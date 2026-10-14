"""The fast backends' FP32 forgetting logits."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from olmo_core.exceptions import OLMoConfigurationError


class ForgettingParameters(nn.Module):
    """FP32 forgetting logits of the embedding write and of every block's two writes."""

    def __init__(
        self,
        *,
        groups: int,
        n_layers: int,
        lambda_init: float,
        learn_lambda: bool,
        init_device: str,
    ) -> None:
        super().__init__()
        if groups <= 0 or n_layers <= 0:
            raise OLMoConfigurationError(
                "RLS forgetting parameters require positive groups and layer count"
            )
        self.groups = groups
        self.n_layers = n_layers
        self.lambda_init = lambda_init
        requires_grad = learn_lambda and lambda_init < 1.0
        self.root_logit = nn.Parameter(
            torch.empty(groups, device=init_device, dtype=torch.float32),
            requires_grad=requires_grad,
        )
        self.block_logits = nn.Parameter(
            torch.empty(
                n_layers,
                2,
                groups,
                device=init_device,
                dtype=torch.float32,
            ),
            requires_grad=requires_grad,
        )

    def reset_parameters(
        self, *, generator: Optional[torch.Generator] = None
    ) -> None:
        del generator
        from ..components import forgetting_logit_init
        value = forgetting_logit_init(self.lambda_init)
        nn.init.constant_(self.root_logit, value)
        nn.init.constant_(self.block_logits, value)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        # FSDP registers hooks on module outputs; fresh zero-copy views keep
        # those hooks off the persistent parameter objects across iterations.
        return (
            self.root_logit.view_as(self.root_logit),
            self.block_logits.view_as(self.block_logits),
        )

    def apply_fsdp(self, dp_mesh: Optional[DeviceMesh]) -> None:
        """Shard the forgetting logits as their own FP32 FSDP unit."""

        fully_shard(
            self,
            mesh=dp_mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
            ),
            reshard_after_forward=False,
        )

    def apply_ddp(self, dp_mesh: Optional[DeviceMesh]) -> None:
        """Replicate the forgetting logits as their own FP32 group."""

        try:
            from torch.distributed._composable.replicate_with_fsdp import replicate
        except ImportError as exc:
            raise OLMoConfigurationError(
                "DDP mixed precision requires PyTorch's replicate_with_fsdp API "
                "to keep the forgetting logits in FP32"
            ) from exc

        replicate(
            self,
            mesh=dp_mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
            ),
        )
