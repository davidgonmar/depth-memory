import logging
from collections import defaultdict
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from olmo_core.data.utils import get_cumulative_document_lengths
from olmo_core.distributed.parallel import get_pp_mesh
from olmo_core.distributed.utils import (
    hide_from_torch,
    install_fsdp_input_cast_refcycle_fix,
    unhide_from_torch,
)
from olmo_core.doc_utils import beta_feature
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.utils import get_default_device, mark_dynamic, move_to_device

from ..attention import (
    Attention,
    FusedAttention,
    RingAttentionLoadBalancer,
    SequenceMixer,
)
from ..buffer_cache import BufferCache
from ..functional import l2_normalize
from ..hyper_connections import (
    MHC,
    HyperConnectionStreamExpand,
    MHCController,
    MHCLearnedStreamReduce,
    MHCParameterBank,
)
from ..layer_norm import LayerNormConfig, RMSNorm
from ..lm_head import LMHeadConfig, LMOutputWithLoss
from ..moe import MoEBase
from ..rope import RoPEBuffers, RotaryEmbeddingBase
from ..utils import selective_checkpointing_context_fn
from .block import (
    MHCTransformerBlock,
    MoETransformerBlock,
    NormalizedTransformerBlock,
    TransformerBlock,
    TransformerBlockBase,
    _attnres_num_sources,
    _attnres_routing_flops_per_token,
)
from .config import (
    TransformerActivationCheckpointingMode,
    TransformerBlockConfig,
    TransformerDataParallelWrappingStrategy,
    resolve_block_configs,
)
from .init import InitMethod

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = [
    "Transformer",
    "NormalizedTransformer",
    "MoETransformer",
    "TransformerDataParallelWrappingStrategy",
    "TransformerActivationCheckpointingMode",
]


log = logging.getLogger(__name__)


class Transformer(nn.Module):
    """
    A typical "Llama-style" transformer implementation.

    :param d_model: The model dimensionality.
    :param vocab_size: The vocab size.
    :param n_layers: The number of transformer layers/blocks.
    :param block: The block configuration. Can be a single block config or a dict of named blocks.
    :param layer_norm: The layer norm config for the final layer norm.
    :param bias: Whether to use a bias in the final linear layer.
    :param dtype: The datatype to use for the linear output layer.
    :param init_device: The device used when initializing parameters.
    :param init_seed: The seed used when initializing parameters.
    :param init_std: The standard deviation used when initializing parameters.
    :param embedding_init_std: The standard deviation used when initializing the embeddings.
    :param block_overrides: Overrides for specific blocks. Not supported if `block` is a dict of named blocks.
    :param block_pattern: The pattern of blocks to use. Required if `block` is a dict of named blocks.
    :param embed_scale: The scale factor for the embeddings.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: TransformerBlockConfig | dict[str, TransformerBlockConfig],
        lm_head: LMHeadConfig,
        embedding_norm: Optional[LayerNormConfig] = None,
        dtype: torch.dtype = torch.float32,
        init_method: InitMethod = InitMethod.normal,
        init_device: str = "cpu",
        init_seed: int = 0,
        init_std: float = 0.02,
        embedding_init_std: Optional[float] = None,
        block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None,
        block_pattern: Optional[List[str]] = None,
        embed_scale: Optional[float] = None,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()

        cache = BufferCache()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dtype = dtype
        self.embed_scale = embed_scale
        self.tie_word_embeddings = tie_word_embeddings

        self.embeddings = nn.Embedding(vocab_size, d_model, dtype=dtype, device=init_device)
        self.embedding_norm = (
            None
            if embedding_norm is None
            else embedding_norm.build(
                d_model,
                init_device=init_device,
            )
        )

        block_configs: List[TransformerBlockConfig] = resolve_block_configs(
            n_layers=n_layers,
            block=block,
            block_pattern=block_pattern,
            block_overrides=block_overrides,
        )

        self.blocks = nn.ModuleDict()
        for block_idx in range(n_layers):
            self.blocks[str(block_idx)] = self._validate_block(
                block_configs[block_idx].build(
                    d_model=d_model,
                    block_idx=block_idx,
                    n_layers=n_layers,
                    init_device=init_device,
                    cache=cache,
                    init_std=init_std,
                )
            )

        hyper_blocks = [
            block for block in self.blocks.values() if isinstance(block, MHCTransformerBlock)
        ]
        self._hyper_connection_enabled = bool(hyper_blocks)
        self._mhc_recompute_layers = 0
        self._mhc_upstream_projection_backward = False
        self._hyper_connection_num_streams = 1
        if self._hyper_connection_enabled:
            if len(hyper_blocks) != n_layers:
                raise OLMoConfigurationError(
                    "mHC residual routing must be configured for all transformer blocks"
                )
            num_streams = {block.hyper_connection_num_streams for block in hyper_blocks}
            if len(num_streams) != 1:
                raise OLMoConfigurationError("All mHC blocks must use the same stream count")
            self._hyper_connection_num_streams = num_streams.pop()
            recompute = {block.hyper_connection_recompute_layers for block in hyper_blocks}
            if len(recompute) != 1:
                raise OLMoConfigurationError("mHC recompute_layers must be the same for all blocks")
            self._mhc_recompute_layers = recompute.pop()
            self._mhc_upstream_projection_backward = any(
                block.hyper_connection_upstream_projection_backward for block in hyper_blocks
            )

        self.expand_residual_streams = HyperConnectionStreamExpand(
            self._hyper_connection_num_streams
        )
        self.reduce_residual_streams: Optional[MHCLearnedStreamReduce] = None
        if self._hyper_connection_enabled:
            assert lm_head.layer_norm is not None
            self.reduce_residual_streams = MHCLearnedStreamReduce(
                self._hyper_connection_num_streams, d_model,
                eps=lm_head.layer_norm.eps, init_device=init_device,
                upstream_projection_backward=self._mhc_upstream_projection_backward,
            )
        self.lm_head = lm_head.build(
            d_model=d_model, vocab_size=vocab_size, init_device=init_device
        )
        if self._hyper_connection_enabled:
            final_norm = self.lm_head.norm
            if (
                final_norm is None
                or not isinstance(final_norm, RMSNorm)
                or final_norm.weight is None
            ):
                raise OLMoConfigurationError(
                    "mHC requires an affine final RMSNorm in the LM head"
                )
        # Bank every mHC controller's and the head's FP32 parameters into one model-owned
        # bank (one FSDP unit; see MHCParameterBank).
        self.mhc_parameter_bank: Optional[MHCParameterBank] = None
        if hyper_blocks and all(block.hyper_connection_parameter_bank for block in hyper_blocks):
            bank = MHCParameterBank(init_device=init_device)
            owners: list[tuple[str, nn.Module]] = []
            for block_key, block in self.blocks.items():
                for module in block.modules():
                    if isinstance(module, MHCController):
                        owners.append((str(block_key), module))
            if isinstance(self.reduce_residual_streams, MHCLearnedStreamReduce):
                owners.append(("head", self.reduce_residual_streams))
            # Reservation order keeps every slot 16-byte aligned for the TileLang kernels: the
            # projections and RMS weights (multiples of 64 elements) first, then the 24-element
            # biases (96 bytes each), then the 4-element head bias, then the 3-element gating
            # factors and the head scale, which the kernels read element-wise.
            for names in (("fn", "weight", "norm_weight"), ("b",), ("bias",), ("alpha",), ("scale",)):
                for owner, module in owners:
                    module.bank_into(bank, owner, names)
            self.mhc_parameter_bank = bank.seal()
            self._mhc_banked_modules = [module for _, module in owners]
        if self.tie_word_embeddings:
            if self.lm_head.w_out.bias is not None:
                raise OLMoConfigurationError(
                    "Tied token embeddings require a bias-free language-model head"
                )
            if self.lm_head.w_out.weight.shape != self.embeddings.weight.shape:
                raise OLMoConfigurationError(
                    "Token embedding and language-model head shapes must match for weight tying"
                )
            self.lm_head.w_out.weight = self.embeddings.weight

        self._attnres_enabled = any(
            getattr(b, "attnres_block_size", None) is not None for b in self.blocks.values()
        )
        self._ldm_config: Optional[Any] = None
        if self._hyper_connection_enabled and self._attnres_enabled:
            raise OLMoConfigurationError(
                "mHC residual routing cannot be combined with AttnRes"
            )
        if self._attnres_enabled:
            final_norm = self.lm_head.norm
            if (
                final_norm is None
                or not isinstance(final_norm, RMSNorm)
                or final_norm.weight is None
            ):
                raise OLMoConfigurationError(
                    "AttnRes requires an affine final RMSNorm in the LM head"
                )
            self.res_proj = nn.Linear(
                d_model,
                1,
                bias=False,
                device=init_device,
                dtype=final_norm.weight.dtype,
            )
            self.res_norm = RMSNorm(
                size=d_model,
                eps=final_norm.eps,
                elementwise_affine=True,
                bias=False,
                full_precision=final_norm.full_precision,
                dtype=final_norm.weight.dtype,
                init_device=init_device,
            )

        ldm_blocks = [
            block for block in self.blocks.values() if hasattr(block, "ldm_rule")
        ]
        self._ldm_enabled = bool(ldm_blocks)
        if self._ldm_enabled:
            if len(ldm_blocks) != n_layers:
                raise OLMoConfigurationError(
                    "LDM must be configured for all transformer blocks"
                )
            if self._hyper_connection_enabled or self._attnres_enabled:
                raise OLMoConfigurationError(
                    "LDM cannot be combined with mHC or AttnRes"
                )
            first_block = cast(TransformerBlock, ldm_blocks[0])
            ldm_config = first_block.ldm
            if any(
                block.ldm != ldm_config
                for block in ldm_blocks[1:]
            ):
                raise OLMoConfigurationError(
                    "All blocks in one LDM segment must use the same complete config"
                )
            self._ldm_config = ldm_config
            from olmo_core.nn.ldm.integration import (
                build_boundaries,
                build_forgetting_parameters,
            )

            (
                self.ldm_boundary,
                self.ldm_output_boundary,
            ) = build_boundaries(
                config=ldm_config,
                d_model=d_model,
                first_block=first_block,
                init_device=init_device,
                dtype=dtype,
                init_std=init_std,
            )
            self.ldm_retention = build_forgetting_parameters(
                config=ldm_config,
                n_layers=n_layers,
                init_device=init_device,
            )

        self.init_device = init_device
        self.init_method = InitMethod(init_method)
        self.init_seed = init_seed
        self.init_std = init_std
        self.embedding_init_std = embedding_init_std

        self._cache = cache
        self._pp_enabled = False
        self._pp_group_size = 1
        self._fp8_enabled = False
        self._precompute_float8_dynamic_scale_for_fsdp = False
        self._compile_enabled = False
        self._device: Optional[torch.device] = None
        self._cp_load_balancer: Optional[RingAttentionLoadBalancer] = None
        self._tp_enabled = False
        self._tp_mesh: Optional[DeviceMesh] = None
        self._fsdp_enabled = False
        # Cache the value of these properties up-front in case the parameters are removed
        # later, like for pipeline parallelism.
        self.num_params
        self.num_non_embedding_params

    def _validate_block(self, block: TransformerBlockBase) -> TransformerBlockBase:
        return block

    def compute_auxiliary_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        del reset
        return {}

    def reset_auxiliary_metrics(self):
        pass

    def reset_attnres_parameters(self) -> None:
        if not self._attnres_enabled:
            return
        nn.init.zeros_(self.res_proj.weight)
        for block in self.blocks.values():
            reset_attnres_parameters = getattr(block, "reset_attnres_parameters", None)
            if reset_attnres_parameters is not None:
                reset_attnres_parameters()

    def reset_mhc_parameters(
        self, *, generator: Optional[torch.Generator] = None
    ) -> None:
        if not self._hyper_connection_enabled:
            return
        from contextlib import nullcontext

        bank = self.mhc_parameter_bank
        with bank.initializing() if bank is not None else nullcontext():
            for module in self.modules():
                if isinstance(module, MHC):
                    module.reset_parameters(generator=generator)
            # Initialize the head LAST to preserve the backbone and connector RNG sequence.
            if isinstance(self.reduce_residual_streams, MHCLearnedStreamReduce):
                self.reduce_residual_streams.reset_parameters(generator=generator)

    def reset_ldm_parameters(
        self, *, generator: Optional[torch.Generator] = None
    ) -> None:
        if not self._ldm_enabled:
            return
        self.ldm_boundary.reset_parameters(generator=generator)
        self.ldm_output_boundary.reset_parameters(generator=generator)
        self.ldm_retention.reset_parameters(generator=generator)
        for block in self.blocks.values():
            rule = getattr(block, "ldm_rule", None)
            if rule is not None:
                rule.reset_parameters(generator=generator)

    @property
    def pp_enabled(self) -> bool:
        return self._pp_enabled

    @property
    def fp8_enabled(self) -> bool:
        return self._fp8_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    @property
    def fsdp_enabled(self) -> bool:
        return self._fsdp_enabled

    @property
    def is_moe(self) -> bool:
        return False

    @property
    def device(self) -> torch.device:
        if self._device is None:
            for p in self.parameters():
                if p.numel() > 0:
                    self._device = p.device
                    break
            else:
                self._device = get_default_device()
        return self._device

    @property
    def compile_enabled(self) -> bool:
        return self._compile_enabled

    def get_rope_buffers(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> Dict[int, Optional[RoPEBuffers]]:
        """
        Get the RoPE buffers to pass to each layer.
        """
        if device is None:
            device = self.device
        rope_buffers = {}
        for key, block in self.blocks.items():
            if isinstance(block.attention, (Attention, FusedAttention)):
                rope = cast(Optional[RotaryEmbeddingBase], block.attention.rope)
                rope_buffers[int(key)] = None if rope is None else rope.get_buffers(seq_len, device)
            else:
                rope_buffers[int(key)] = None
        return rope_buffers

    def to_empty(self, *, device: torch.device | str, recurse: bool = True):
        """Materialize parameters while preserving configured shared identities."""

        result = super().to_empty(device=device, recurse=recurse)
        if self.tie_word_embeddings:
            if self.lm_head.w_out.bias is not None:
                raise OLMoConfigurationError(
                    "Tied token embeddings require a bias-free language-model head"
                )
            if self.lm_head.w_out.weight.shape != self.embeddings.weight.shape:
                raise OLMoConfigurationError(
                    "Token embedding and language-model head shapes must match for weight tying"
                )
            self.lm_head.w_out.weight = self.embeddings.weight
        return result

    @torch.no_grad()
    def init_weights(
        self,
        *,
        max_seq_len: Optional[int] = None,
        max_local_microbatch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        world_mesh: Optional[DeviceMesh] = None,
    ) -> torch.Generator:
        """
        Initialize the model weights.

        :param max_seq_len: The maximum sequence length expected. This is used
            to warm up the RoPE cache.
        :param max_local_microbatch_size: The maximum local (rank) micro-batch size (in tokens)
            expected. This is used to warm-up some MoE cache.
        :param device: The device the local copy of the model will be trained on.
        """
        device = device or self.device
        self.to_empty(device=device)

        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()  # type: ignore

        seed = self.init_seed
        if world_mesh is not None and self.pp_enabled:
            seed += get_pp_mesh(world_mesh).get_local_rank()

        generator = torch.Generator(device).manual_seed(seed)

        if self.embeddings is not None:
            self.init_method.init_embeddings(
                self.embeddings,
                d_model=self.d_model,
                embed_scale=self.embed_scale,
                std=self.embedding_init_std
                if self.embedding_init_std is not None
                else self.init_std,
                generator=generator,
            )

        for block in self.blocks.values():
            # This might fail if it's wrapped.
            #  assert isinstance(block, TransformerBlock)
            block = cast(TransformerBlock, block)
            att = cast(SequenceMixer, block.attention)

            # Attention weights.
            self.init_method.init_attention(
                att,
                d_model=self.d_model,
                block_idx=block.block_idx,
                num_blocks=self.n_layers,
                std=self.init_std,
                generator=generator,
            )

            # Feed-forward weights.
            if hasattr(block, "feed_forward"):
                self.init_method.init_feed_forward(
                    block.feed_forward,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )

            # MoE weights.
            if hasattr(block, "feed_forward_moe"):
                block = cast(MoETransformerBlock, block)
                if max_local_microbatch_size is not None:
                    block.feed_forward_moe.warmup_cache(max_local_microbatch_size)
                self.init_method.init_feed_forward_moe(
                    block.feed_forward_moe,
                    d_model=self.d_model,
                    block_idx=block.block_idx,
                    num_blocks=self.n_layers,
                    std=self.init_std,
                    generator=generator,
                )

            if isinstance(att, (Attention, FusedAttention)):
                # Warm up attention backend cache.
                if max_seq_len is not None and att.backend is not None:
                    att.backend.warmup_cache(max_seq_len, device)

                # Warm up RoPE cache.
                if max_seq_len is not None and att.rope is not None:
                    att.rope.warmup_cache(max_seq_len, device)

        if self.lm_head is not None:
            self.init_method.init_final_w_out(
                self.lm_head.w_out,
                d_model=self.d_model,
                std=self.init_std,
                generator=generator,
            )

        self.reset_attnres_parameters()
        self.reset_ldm_parameters(generator=generator)
        # mHC last: the controllers' initialization is constant, and the learned head draws from
        # the model generator after the backbone.
        self.reset_mhc_parameters(generator=generator)

        return generator

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        *,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Dict[str, Any],
        Dict[int, Dict[str, Any]],
        Dict[str, Any],
    ]:
        # NOTE: with pipeline parallelism input_ids might actually be an intermediate output,
        # so we have to be careful here.
        B, S = input_ids.shape[:2]

        all_block_kwargs: Dict[str, Any] = {}
        per_block_kwargs: Dict[int, Dict[str, Any]] = defaultdict(dict)
        lm_head_kwargs: Dict[str, Any] = dict(
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            return_logits=return_logits,
            logits_to_keep=logits_to_keep,
        )

        if loss_div_factor is not None:
            loss_div_factor = move_to_device(loss_div_factor, self.device)
            lm_head_kwargs["loss_div_factor"] = loss_div_factor
            all_block_kwargs["loss_div_factor"] = loss_div_factor

        # Prepare document length inputs.
        max_doc_len: Optional[int] = None
        cu_doc_lens: Optional[torch.Tensor] = None
        doc_lens: Optional[torch.Tensor] = None
        cache_leftpad: Optional[torch.Tensor] = kwargs.pop("cache_leftpad", None)

        if (doc_lens := kwargs.pop("doc_lens", None)) is not None and (
            max_doc_lens := kwargs.pop("max_doc_lens", None)
        ) is not None:
            max_doc_len = max(max_doc_lens)
            cu_doc_lens = get_cumulative_document_lengths(doc_lens)

        # Shard inputs and RoPE buffers on sequence dimension if using context parallelism.
        if (cp_load_balancer := self._cp_load_balancer) is not None:
            inputs = [input_ids]
            seq_dims = [1]
            pad_values: List[Union[int, float]] = [0]
            keys = ["input_ids"]

            # NOTE: initialize buffer(s) on CPU to avoid possible host-device sync when sharding.
            for block_idx, rope_buffers in self.get_rope_buffers(S, torch.device("cpu")).items():
                if rope_buffers is not None:
                    # Also shard RoPE buffers based on the context parallelism load balancer.
                    if rope_buffers.pos_sin is not None:
                        inputs.append(rope_buffers.pos_sin)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.pos_sin")
                    if rope_buffers.pos_cos is not None:
                        inputs.append(rope_buffers.pos_cos)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.pos_cos")
                    if rope_buffers.freqs_cis is not None:
                        inputs.append(rope_buffers.freqs_cis)
                        seq_dims.append(0)
                        pad_values.append(0.0)
                        keys.append(f"block_{block_idx}.freqs_cis")

            if labels is not None:
                inputs.append(labels)
                seq_dims.append(1)
                pad_values.append(ignore_index)
                keys.append("labels")

            if cache_leftpad is not None:
                raise NotImplementedError("cache_leftpad is not supported with context parallelism")

            if cu_doc_lens is not None:
                # NOTE: Can only shard properly here if 'input_ids' is flat, i.e. a single instance.
                # TODO: (epwalsh) We could just flatten all of the inputs here, but then we risk going
                # beyond the model's maximum sequence length, which might be okay at least
                # with relative positional encodings, but then again if you're resorting to context
                # parallelism you can probably only fit a single instance at a time anyway.
                if B != 1:
                    raise RuntimeError(
                        f"Rank micro-batches must consist of a single instance when using "
                        f"context parallelism with intra-document masking (got {B} instances)"
                    )
                inputs, additional_inputs = cp_load_balancer.batch_shard_by_document(
                    inputs=inputs,
                    seq_dims=seq_dims,
                    cu_doc_lens=cu_doc_lens,
                    pad_values=pad_values,
                    length_multiple=16,
                )
                for key, value in additional_inputs.items():
                    all_block_kwargs[key] = move_to_device(value, self.device)

            else:
                inputs = cp_load_balancer.batch_shard(
                    inputs=inputs,
                    seq_dims=seq_dims,
                    pad_values=pad_values,
                )

            for key, value in zip(keys, inputs):
                if key.startswith("block_"):
                    block_key, key = key.split(".", 1)
                    block_idx = int(block_key.replace("block_", ""))
                    per_block_kwargs[block_idx][key] = move_to_device(value, self.device)
                else:
                    all_block_kwargs[key] = move_to_device(value, self.device)

            input_ids = all_block_kwargs.pop("input_ids")
            labels = all_block_kwargs.pop("labels", None)
        else:
            input_ids = move_to_device(input_ids, self.device)
            labels = move_to_device(labels, self.device)

            if (max_doc_len is not None or cu_doc_lens is not None) and cache_leftpad is not None:
                raise ValueError("max_doc_len/cu_doc_lens and cache_leftpad are mutually exclusive")
            if max_doc_len is not None or cu_doc_lens is not None:
                all_block_kwargs["max_doc_len"] = max_doc_len
                all_block_kwargs["cu_doc_lens"] = move_to_device(cu_doc_lens, self.device)
            if cache_leftpad is not None:
                all_block_kwargs["cache_leftpad"] = move_to_device(cache_leftpad, self.device)

        if "cu_doc_lens" in all_block_kwargs:
            mark_dynamic(all_block_kwargs["cu_doc_lens"], 0, strict=False)  # type: ignore[arg-type]

        return (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """
        Run the transformer on the token input IDs.

        :param input_ids: The token input IDs, shape ``(batch_size, seq_len)``.
        :param labels: The token labels, shape ``(batch_size, seq_len)``.
        :param ignore_index: The index to ignore in the loss computation. Default is -100.
        :param loss_reduction: The reduction method for the loss. Can be "mean", "sum", or "none".
        :param z_loss_multiplier: Optional multiplier for the z-loss regularization term.
        :param loss_div_factor: Optional divisor for the loss, can be a scalar or tensor.
        :param return_logits: Whether to return logits along with the loss when labels are provided.
        :param logits_to_keep: Number of positions to keep from the end of the sequence (if int),
            or tensor specifying which positions to keep. Default is 0 (keep all).

        :returns: The logits if ``labels`` is ``None`` or the losses if ``labels`` is not ``None``.
        """
        (
            input_ids,
            labels,
            all_block_kwargs,
            per_block_kwargs,
            lm_head_kwargs,
        ) = self._prepare_inputs(
            input_ids,
            labels,
            ignore_index=ignore_index,
            loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier,
            loss_div_factor=loss_div_factor,
            return_logits=return_logits,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # Get embeddings but pass-through for non-existent layers to allow easy
        # pipeline parallel configuration.
        h = self.embeddings(input_ids) if self.embeddings is not None else input_ids
        if self.embeddings is not None and self.embed_scale is not None:
            h = h * self.embed_scale
        if self.embedding_norm is not None:
            h = self.embedding_norm(h)
        if self._hyper_connection_enabled and self.embeddings is not None:
            h = self.expand_residual_streams(h)
        if self.mhc_parameter_bank is not None:
            self.mhc_parameter_bank()  # FSDP / replicate: unshard the FP32 bank once for all blocks
            for module in self._mhc_banked_modules:
                module.refresh_bank_views()

        attnres_states: Optional[List[torch.Tensor]] = None
        if self._ldm_enabled:
            from olmo_core.nn.ldm.integration import run_segment

            h = run_segment(
                h,
                self.blocks,
                boundary=self.ldm_boundary,
                output_boundary=self.ldm_output_boundary,
                forgetting=self.ldm_retention,
                config=self._ldm_config,
                all_block_kwargs=all_block_kwargs,
                per_block_kwargs=per_block_kwargs,
                compile_enabled=self.compile_enabled,
            )
        else:
            # PreLN, AttnRes and mHC run block by block; an LDM backend owns its whole segment.
            # mHC recompute groups (section 4.3.2 of the mHC paper): L_r mHC layers, that is
            # L_r / 2 blocks, share one entry residual. Each group is a plain object handed to its
            # blocks; the ops inside rebuild the streams from it in backward (see
            # hyper_connections.MHCGroup).
            from ..hyper_connections import MHCGroup

            group_blocks = self._mhc_recompute_layers // 2 if torch.is_grad_enabled() else 0
            mhc_group = None
            n_blocks = len(self.blocks)
            for ordinal, (block_key, block) in enumerate(self.blocks.items()):
                block_idx = int(block_key)
                if group_blocks and ordinal % group_blocks == 0:
                    mhc_group = MHCGroup(
                        h, self._hyper_connection_num_streams,
                        n_layers=2 * min(group_blocks, n_blocks - ordinal),
                    )
                block_kwargs = per_block_kwargs.get(block_idx, {})
                if mhc_group is not None:
                    block_kwargs = {**block_kwargs, "mhc_group": mhc_group}
                if self.compile_enabled:
                    mark_dynamic(h, (0, 1), strict=False)
                if self._attnres_enabled:
                    h, attnres_states = block(
                        h,
                        attnres_states=attnres_states,
                        **all_block_kwargs,
                        **block_kwargs,
                    )
                else:
                    h = block(h, **all_block_kwargs, **block_kwargs)

        if self._attnres_enabled:
            from olmo_core.kernels.attnres import fused_attnres

            residuals = [*attnres_states, h]
            h = fused_attnres(
                query=self.res_proj.weight,
                residuals=residuals,
                rms_weight=self.res_norm.weight,
                key_rms_eps=self.res_norm.eps,
            )
        # Get final logits but again pass-through in case of pipeline parallelism.
        if self.lm_head is not None:
            if self.reduce_residual_streams is not None:
                h = self.reduce_residual_streams(h)
                if self.mhc_parameter_bank is not None:
                    for module in self._mhc_banked_modules:
                        module.release_bank_views()  # the unsharded bank is only valid during forward
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
                if labels is not None:
                    mark_dynamic(labels, (0, 1), strict=False)
            # NOTE: When TP is active we can't pass 'labels=None' or the hook from 'PrepareModuleInput'
            # will throw an exception.
            if labels is not None:
                lm_head_kwargs["labels"] = labels
            return self.lm_head(h, **lm_head_kwargs)
        else:
            return h

    def apply_fp8(self, float8_config: Float8Config):
        """
        Use an FP8 recipe on most linear layers.
        """
        if not float8_config.enabled:
            return

        modules_to_ignore = set()
        if self.lm_head is not None:
            modules_to_ignore.add("lm_head.w_out")
        if float8_config.modules_to_ignore is not None:
            modules_to_ignore.update(float8_config.modules_to_ignore)

        float8_config.apply_float8_linear(self, modules_to_ignore=modules_to_ignore)

        self._fp8_enabled = True
        self._precompute_float8_dynamic_scale_for_fsdp = (
            float8_config.should_precompute_float8_dynamic_scale_for_fsdp
        )

    def apply_pp(self, pp_mesh: DeviceMesh):
        """
        Prepare the model for pipeline parallelism after it's been split into stages.
        """
        if self._ldm_enabled:
            raise OLMoConfigurationError(
                "LDM models do not support pipeline parallelism: the depth memory state "
                "segment must stay within one model stage"
            )
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_pp(pp_mesh)
        self._pp_enabled = True
        self._pp_group_size = pp_mesh.size()

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: Optional[bool] = None):
        """
        Apply tensor parallelism to the model.

        :param loss_parallel: Set to ``True`` if parallelizing the loss function as well.
        :param float8_enabled: Set this to ``True`` if training with float8 linear layers.
        """
        if self._ldm_enabled:
            raise OLMoConfigurationError(
                "LDM models do not support tensor parallelism: DTensor sharding of the state "
                "and custom ops is not implemented"
            )
        if float8_enabled is None:
            float8_enabled = self.fp8_enabled
        elif not float8_enabled and self.fp8_enabled:
            raise OLMoConfigurationError(
                "Got 'float8_enabled=False', but FP8 has already been enabled"
            )

        if self.embeddings is not None:
            parallelize_module(
                self.embeddings,
                device_mesh=tp_mesh,
                parallelize_plan=RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                    use_local_output=False,
                ),
            )
        if self.embedding_norm is not None:
            parallelize_module(
                self.embedding_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
            )

        # Apply tensor/sequence parallelism to every transformer block.
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_tp(tp_mesh, input_layout=Shard(1), float8_enabled=float8_enabled)

        if self.lm_head is not None:
            self.lm_head.apply_tp(tp_mesh, input_layouts=(Shard(1), Replicate()))

        self._tp_enabled = True
        self._tp_mesh = tp_mesh

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: RingContextParallelStyle | None = None,
        uly: UlyssesContextParallelStyle | None = None,
    ):
        """
        Prepare the model for context-parallelism (CP).

        :param cp_mesh: The CP device mesh.
        :param ring: The ring context parallel style.
        :param uly: The ulysses context parallel style.
        """
        if ring is not None:
            self._cp_load_balancer = ring.load_balancer.build(cp_mesh)
        elif uly is not None:
            self._cp_load_balancer = uly.load_balancer.build(cp_mesh)

        for block in self.blocks.values():
            cast(TransformerBlockBase, block).apply_cp(cp_mesh, ring=ring, uly=uly)
        if self.lm_head is not None:
            self.lm_head.apply_cp(cp_mesh)

    def apply_activation_checkpointing(
        self,
        mode: TransformerActivationCheckpointingMode,
        block_interval: Optional[int] = None,
        modules: Optional[List[str]] = None,
        activation_memory_budget: Optional[float] = None,
    ):
        """
        Apply activation checkpointing to the model.

        :param mode: Determines how to apply activation checkpointing.
        :param block_interval: Required when :data:`mode` is "selected_blocks". Determines
            which blocks are wrapped.
        :param modules: Required when :data:`mode` is "selected_modules". A list of modules names
            to wrap for activation checkpointing. Globs are supported.
        :param activation_memory_budget: The memory budget for activation checkpointing in the range
            [0, 1]. 0 corresponds to the memory usage when recomputing all activations, and 1
            corresponds to the memory usage when recomputing no activations (which is the default).
            Requires compilation to be enabled.
        """

        if mode == TransformerActivationCheckpointingMode.budget:
            if activation_memory_budget is None:
                raise ValueError("'activation_memory_budget' is required for 'budget' mode")
            if activation_memory_budget < 0 or activation_memory_budget > 1:
                raise ValueError("'activation_memory_budget' must be in the range [0, 1]")
            torch._functorch.config.activation_memory_budget = activation_memory_budget
            return

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper as ptd_checkpoint_wrapper,
        )

        if (
            mode == TransformerActivationCheckpointingMode.selected_blocks
            and block_interval is None
        ):
            raise ValueError("'block_interval' is required for 'selected_blocks' mode")

        if mode == TransformerActivationCheckpointingMode.selected_modules and modules is None:
            raise ValueError("'modules' is required for 'selected_modules' mode")

        # TODO: only preserve RNG state if dropout is active
        preserve_rng_state = False

        if mode == TransformerActivationCheckpointingMode.selected_modules:
            from fnmatch import fnmatch

            assert modules is not None
            wrapped_modules: Set[str] = set()
            for name, module in self.named_modules():
                for pattern in modules:
                    if fnmatch(name, pattern):
                        break
                else:
                    continue

                if isinstance(module, MoEBase):
                    raise OLMoConfigurationError(
                        "Wrapping an entire MoE module for activation checkpointing is not supported. "
                        "Please try a finer-grained wrapping strategy."
                    )

                # NOTE: have to be careful not to try to wrap submodules of modules that have been wrapped.
                parent_name = ".".join(name.split(".")[:-1])
                if parent_name in wrapped_modules:
                    continue

                parent = self if not parent_name else self.get_submodule(parent_name)
                module = ptd_checkpoint_wrapper(module, preserve_rng_state=preserve_rng_state)
                parent.register_module(name.split(".")[-1], module)
                log.info(f"Wrapped '{name}' for activation checkpointing")
                wrapped_modules.add(name)
        else:
            for block_idx, block in enumerate(self.blocks.values()):
                if mode == TransformerActivationCheckpointingMode.selected_blocks:
                    assert block_interval is not None
                    if block_idx % block_interval == 0:
                        if isinstance(block, MoETransformerBlock):
                            raise OLMoConfigurationError(
                                "Wrapping MoE blocks for activation checkpointing is not supported."
                            )
                        block = ptd_checkpoint_wrapper(block, preserve_rng_state=preserve_rng_state)
                elif mode == TransformerActivationCheckpointingMode.full:
                    if isinstance(block, MoETransformerBlock):
                        raise OLMoConfigurationError(
                            "Wrapping MoE blocks for activation checkpointing is not supported."
                        )
                    block = ptd_checkpoint_wrapper(block, preserve_rng_state=preserve_rng_state)
                elif mode == TransformerActivationCheckpointingMode.selected_ops:
                    block = ptd_checkpoint_wrapper(
                        block,
                        context_fn=selective_checkpointing_context_fn,
                        preserve_rng_state=preserve_rng_state,
                    )

                self.blocks.register_module(str(block_idx), block)

    def apply_compile(self):
        """
        Apply ``torch.compile()`` to each transformer block, which makes compilation efficient
        due to repeated structure.

        .. warning::
            This must be called after :meth:`apply_activation_checkpointing()` but
            before :meth:`apply_fsdp()` or :meth:`apply_ddp()`.
        """
        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_compile()

        if isinstance(self.reduce_residual_streams, MHCLearnedStreamReduce):
            self.reduce_residual_streams.compile(fullgraph=False, dynamic=False)

        if self.lm_head is not None:
            self.lm_head.compile(
                fullgraph=False,
                dynamic=False,
            )

        torch.compiler.config.dynamic_sources += "L['kwargs']['max_doc_len'],"
        self._compile_enabled = True

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
    ):
        """
        Apply FSDP(2) to the model.

        .. warning::
            This should generally be called last if using any other parallelism strategies or optimizations
            like :meth:`apply_compile()`.

        :param dp_mesh: The model data parallel device mesh.
        :param param_dtype: The data type to materialize params in. Defaults to the current param dtype.
        :param reduce_dtype: The data type for gradient reduction.
        :pp_enabled: If pipeline parallelism is also enabled.
        :prefetch_factor: For tuning the prefetch settings. 0 is the default, and higher values result
            in more aggressive prefetching.
        :wrapping_strategy: The wrapping strategy.
        """
        if self._mhc_recompute_layers and wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            raise OLMoConfigurationError(
                "mHC recompute_layers > 0 requires the 'full' (or 'blocks') FSDP wrapping strategy: the "
                "branch's first projections run from the hyper-connection wrapper, outside the attention "
                "and feed-forward FSDP units that 'fine_grained' creates, so those units' parameters would "
                "still be sharded there"
            )
        install_fsdp_input_cast_refcycle_fix()
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = dict(mesh=dp_mesh, mp_policy=mp_policy)
        # For PP, do not reshard after forward to avoid per-microbatch all-gathers,
        # which can be expensive and non-overlapped
        reshard_after_forward = False if pp_enabled else True

        if self._ldm_enabled:
            self.ldm_retention.apply_fsdp(dp_mesh)

        for block in self.blocks.values():
            block = cast(TransformerBlockBase, block)
            block.apply_fsdp(
                dp_mesh=dp_mesh,
                prefetch_factor=prefetch_factor,
                wrapping_strategy=wrapping_strategy,
                reshard_after_forward=reshard_after_forward,
                mp_policy=mp_policy,
            )

        if self.embeddings is not None and not self.tie_word_embeddings:
            fully_shard(
                self.embeddings,
                reshard_after_forward=reshard_after_forward,
                **fsdp_config,
            )
            # Embedding params are not needed for backwards computation.
            cast(FSDPModule, self.embeddings).set_unshard_in_backward(False)

        if wrapping_strategy != TransformerDataParallelWrappingStrategy.blocks:
            if self.embedding_norm is not None:
                fully_shard(self.embedding_norm, **fsdp_config)
            if self.lm_head is not None and not self.tie_word_embeddings:
                fully_shard(self.lm_head, reshard_after_forward=False, **fsdp_config)

        if self.mhc_parameter_bank is not None:
            # One FP32 unit for all controllers and the head. It stays unsharded from the bank
            # call at forward start until the end of backward (its reduce-scatter runs in FSDP's
            # root post-backward callback), so no FSDP hook fires inside a block.
            fully_shard(
                self.mhc_parameter_bank, mesh=dp_mesh,
                mp_policy=MHCTransformerBlock._controller_mixed_precision_policy(),
                reshard_after_forward=False,
            )
        elif isinstance(self.reduce_residual_streams, MHCLearnedStreamReduce):
            # With parameter_bank=False the FP32 head is its own unit.
            fully_shard(
                self.reduce_residual_streams, mesh=dp_mesh,
                mp_policy=MHCTransformerBlock._controller_mixed_precision_policy(),
                reshard_after_forward=reshard_after_forward,
            )
        fully_shard(self, reshard_after_forward=reshard_after_forward, **fsdp_config)
        # Some inputs need to be on CPU initially, but FSDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

        if prefetch_factor > 0:
            blocks = cast(List[FSDPModule], list(self.blocks.values()))
            if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
                for i, block in enumerate(blocks):
                    if i + 1 < len(blocks):
                        targets: List[FSDPModule] = blocks[i + 1 : i + 1 + prefetch_factor]
                    elif isinstance(self.lm_head, FSDPModule):
                        targets = [self.lm_head]
                    else:
                        targets = []
                    if not targets:
                        continue

                    last_child = getattr(block, "feed_forward", None)
                    if last_child is None:
                        last_child = getattr(block, "feed_forward_moe", None)
                    if isinstance(last_child, FSDPModule):
                        last_child.set_modules_to_forward_prefetch(targets)
                    else:
                        block.set_modules_to_forward_prefetch(targets)
            else:
                for i in range(len(blocks)):
                    block = blocks[i]
                    if i + 1 < len(blocks):
                        block.set_modules_to_forward_prefetch(
                            blocks[i + 1 : i + 1 + prefetch_factor]
                        )
                    elif isinstance(self.lm_head, FSDPModule):
                        block.set_modules_to_forward_prefetch([self.lm_head])

        self._fsdp_enabled = True

    def apply_ddp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
    ):
        """
        Replicate the model for data-parallel training without parameter sharding.

        This uses FSDP2's replicate-only implementation so the authoritative
        parameters and optimizer state stay in their original dtype while
        forward/backward use ``param_dtype`` working parameters and gradients
        reduce in ``reduce_dtype``. The transformation is in-place, preserving
        the :class:`Transformer` interface.
        """
        try:
            from torch.distributed._composable.replicate_with_fsdp import replicate
        except ImportError as exc:
            raise OLMoConfigurationError(
                "DDP mixed precision requires PyTorch's replicate_with_fsdp API"
            ) from exc
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        install_fsdp_input_cast_refcycle_fix()

        if self._ldm_enabled:
            # The forgetting logits stay FP32 in their own replicate group.
            self.ldm_retention.apply_ddp(dp_mesh)

        for block in self.blocks.values():
            if isinstance(block, CheckpointWrapper):
                block = block._checkpoint_wrapped_module
            if isinstance(block, MHCTransformerBlock):
                block.apply_ddp(dp_mesh)

        if self.mhc_parameter_bank is not None:
            replicate(
                self.mhc_parameter_bank, mesh=dp_mesh,
                mp_policy=MHCTransformerBlock._controller_mixed_precision_policy(),
            )
        elif isinstance(self.reduce_residual_streams, MHCLearnedStreamReduce):
            replicate(
                self.reduce_residual_streams, mesh=dp_mesh,
                mp_policy=MHCTransformerBlock._controller_mixed_precision_policy(),
            )
        replicate(
            self,
            mesh=dp_mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=param_dtype or self.dtype,
                reduce_dtype=reduce_dtype,
            ),
        )
        # Some inputs need to be on CPU initially, but DDP will move everything to model's
        # device if we don't hide it.
        self.register_forward_pre_hook(_hide_cpu_inputs_from_torch, prepend=True, with_kwargs=True)
        self.register_forward_pre_hook(
            _unhide_cpu_inputs_from_torch, prepend=False, with_kwargs=True
        )

    @cached_property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @cached_property
    def num_non_embedding_params(self) -> int:
        return self.num_params - self.embeddings.weight.numel()

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Returns the idealized number of flops per token for the given sequence length. Purposefully
        does not account for wasted flops due to padding, recomputation, etc.
        """
        flops_per_token = 0
        blocks = cast(List[TransformerBlockBase], list(self.blocks.values()))
        for block in blocks:
            flops_per_token += block.num_flops_per_token(seq_len)
        if self._attnres_enabled:
            # The final aggregation is atomic site 2L. Every block uses the same block size, so
            # read it from the last block.
            final_block = blocks[-1]
            final_block_size = cast(Any, final_block).attnres_block_size
            final_num_sources = _attnres_num_sources(2 * self.n_layers, final_block_size)
            flops_per_token += _attnres_routing_flops_per_token(
                self.d_model, final_num_sources
            )
        if self.lm_head is not None:
            flops_per_token += self.lm_head.num_flops_per_token(seq_len)
        return flops_per_token

    def post_batch(self, dry_run: bool = False):
        """
        Should be called right after the final backward of a complete batch but before the optimizer step.
        """
        del dry_run

    def post_optim_step(self):
        """
        Should be called right after an optimizer step.
        """
        if self.fp8_enabled and self._precompute_float8_dynamic_scale_for_fsdp:
            from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

            precompute_float8_dynamic_scale_for_fsdp(self)


@beta_feature
class NormalizedTransformer(Transformer):
    """
    A nGPT transformer implementation, to be used with the :class:`NormalizedTransformerBlock` block
    type.
    """

    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        n_layers: int,
        block: TransformerBlockConfig | dict[str, TransformerBlockConfig],
        lm_head: LMHeadConfig,
        dtype: torch.dtype = torch.float32,
        init_method: InitMethod = InitMethod.normalized,
        init_device: str = "cpu",
        init_seed: int = 0,
        init_std: float = 0.02,
        embedding_init_std: Optional[float] = None,
        block_overrides: Optional[Dict[int, TransformerBlockConfig]] = None,
        block_pattern: Optional[List[str]] = None,
        tie_word_embeddings: bool = False,
    ):
        super().__init__(
            d_model=d_model,
            vocab_size=vocab_size,
            n_layers=n_layers,
            block=block,
            lm_head=lm_head,
            dtype=dtype,
            init_method=init_method,
            init_device=init_device,
            init_seed=init_seed,
            init_std=init_std,
            embedding_init_std=embedding_init_std,
            block_overrides=block_overrides,
            block_pattern=block_pattern,
            tie_word_embeddings=tie_word_embeddings,
        )

    def _validate_block(self, block: TransformerBlockBase) -> TransformerBlockBase:
        if not isinstance(block, NormalizedTransformerBlock):
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' requires a '{NormalizedTransformerBlock.__name__}' block"
            )
        return block

    @torch.no_grad()
    def init_weights(self, *args, **kwargs) -> torch.Generator:
        generator = super().init_weights(*args, **kwargs)
        self.normalize_matrices()
        return generator

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        if self.embeddings is not None:
            self._normalize_matrix(self.embeddings.weight)

        for block in self.blocks.values():
            if hasattr(block, "normalize_matrices"):
                block.normalize_matrices()  # type: ignore

        if self.lm_head is not None:
            self.lm_head.normalize_matrices()  # type: ignore

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        float8_enabled: Optional[bool] = None,
    ):
        del tp_mesh, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized transformer variant"
        )

    def apply_compile(self):
        super().apply_compile()
        self.normalize_matrices = torch.compile(self.normalize_matrices)

    def post_optim_step(self):
        super().post_optim_step()
        self.normalize_matrices()


@beta_feature
class MoETransformer(Transformer):
    """
    An MoE transformer implementation, to be used with one of the
    :class:`MoETransformerBlock` block types.
    """

    @property
    def is_moe(self) -> bool:
        return True

    def compute_auxiliary_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        mean_offset = 1.0
        if self.pp_enabled:
            # Change the divisor to 'world_size // pp_group_size'
            mean_offset = self._pp_group_size

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}
        for block_idx, block in self.blocks.items():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block_metrics = block.compute_metrics(reset=reset)
            for metric_name, (metric_val, reduce_type) in block_metrics.items():
                out[f"block {int(block_idx):02d}/{metric_name}"] = (
                    metric_val,
                    reduce_type,
                )

                if self.pp_enabled and reduce_type == ReduceType.mean:
                    metric_val = metric_val.float() * mean_offset

                if metric_name not in out:
                    out[metric_name] = (metric_val, reduce_type)
                elif reduce_type in (ReduceType.mean, ReduceType.sum):
                    out[metric_name] = (
                        out[metric_name][0] + metric_val,
                        reduce_type,
                    )
                elif reduce_type == ReduceType.max:
                    out[metric_name] = (
                        torch.max(out[metric_name][0], metric_val),
                        reduce_type,
                    )
                else:
                    raise NotImplementedError(reduce_type)
        return out

    def reset_auxiliary_metrics(self):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            cast(MoETransformerBlock, block).reset_metrics()

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block.apply_ep(ep_mesh, **kwargs)

    def prepare_experts_for_fsdp(
        self,
        world_mesh: DeviceMesh,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: torch.dtype = torch.float32,
        pp_enabled: bool = False,
    ):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            reshard_after_forward = True
            if pp_enabled or block.ep_enabled or block.tp_enabled:
                reshard_after_forward = False
            block.feed_forward_moe.prepare_experts_for_fsdp(
                world_mesh=world_mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=param_dtype or self.dtype, reduce_dtype=reduce_dtype
                ),
                reshard_after_forward=reshard_after_forward,
            )

    def prepare_experts_for_ddp(self, world_mesh: DeviceMesh):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            cast(MoETransformerBlock, block).feed_forward_moe.prepare_experts_for_ddp(
                world_mesh=world_mesh,
            )

    def post_batch(self, dry_run: bool = False):
        for block in self.blocks.values():
            if not block.is_moe:
                continue
            block = cast(MoETransformerBlock, block)
            block.feed_forward_moe.post_batch(dry_run=dry_run)


def _hide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = hide_from_torch(doc_lens)
    return (args, kwargs)


def _unhide_cpu_inputs_from_torch(m, args, kwargs) -> Optional[Tuple[Any, Dict[str, Any]]]:
    del m
    if (doc_lens := kwargs.get("doc_lens")) is not None:
        kwargs["doc_lens"] = unhide_from_torch(doc_lens)
    return (args, kwargs)
