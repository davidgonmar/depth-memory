"""The inference form of an edge: the tensors the kernels read, built once from the trained parameters.

An edge is the write of one site followed by the read of the next. `InferenceEdge.from_sites` turns the
two sites' parameter modules into:

- ``first_layer`` ``[D / 128, 2H, 128]`` BF16: the first layer of both address networks (write rows,
  then read rows) with the context gains folded in as ``BF16(W * g)``, in 128-column tiles;
- ``first_bias`` ``[2H]`` BF16;
- ``second_layer`` ``[2, G, K + 1, H + 8]`` BF16: per site and group the K head rows and the gate row
  over the hidden width, zero padded so that neighbouring rows start in different shared-memory banks;
- ``head_bias`` ``[2, G, K]``, ``gate_bias`` ``[2, G]`` and ``static_address`` ``[2, G, K]`` BF16;
- ``forgetting`` ``[G]`` FP32: the factor that divides the covariance update, ``max(sigmoid(logit), eps)``;
- ``read_gain`` ``[G]`` FP32 holding the BF16-rounded learned gain.

The first index of the tensors of both sites is the site: 0 for the write, 1 for the read.
"""

from __future__ import annotations

import torch
from torch import nn

from .kernels.transforms import forgetting

TILE = 128          # first-layer columns per tile
ROW_PAD = 8         # BF16 padding of a second-layer row


class InferenceEdge(nn.Module):
    """The tensors of one edge as buffers. Inference only: loading training parameters into it raises."""

    def __init__(self, first_layer, first_bias, second_layer, head_bias, gate_bias, static_address,
                 forgetting_factor, gain):
        super().__init__()
        tiles, rows, tile = first_layer.shape
        _, groups, heads, padded = second_layer.shape
        hidden, key_dim = rows // 2, heads - 1
        expected = {"first_layer": (tiles, 2 * hidden, TILE), "first_bias": (2 * hidden,),
                    "second_layer": (2, groups, key_dim + 1, hidden + ROW_PAD), "head_bias": (2, groups, key_dim),
                    "gate_bias": (2, groups), "static_address": (2, groups, key_dim), "forgetting": (groups,),
                    "read_gain": (groups,)}
        tensors = {"first_layer": first_layer, "first_bias": first_bias, "second_layer": second_layer,
                   "head_bias": head_bias, "gate_bias": gate_bias, "static_address": static_address,
                   "forgetting": forgetting_factor, "read_gain": gain}
        for name, tensor in tensors.items():
            dtype = torch.float32 if name in ("forgetting", "read_gain") else torch.bfloat16
            if tuple(tensor.shape) != expected[name] or tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous {dtype} tensor of shape {expected[name]}, "
                                 f"got {tuple(tensor.shape)} {tensor.dtype}")
            self.register_buffer(name, tensor)
        self.groups, self.key_dim, self.hidden = groups, key_dim, hidden

    @classmethod
    @torch.no_grad()
    def from_sites(cls, write_projection, static_key, forgetting_logit, read_projection, static_query,
                   read_gain_parameter, *, groups: int, key_dim: int, eps: float) -> InferenceEdge:
        """The inference form of an edge from the trained modules of its write and read sites."""
        sites = (write_projection, read_projection)
        hidden = write_projection.w1.weight.shape[0]
        model_dim = write_projection.w1.weight.shape[1]
        if model_dim % TILE:
            raise ValueError(f"fast-inference needs a model width that is a multiple of {TILE}, got {model_dim}")
        first_layer = torch.cat([_folded_first_layer(site) for site in sites])
        first_layer = first_layer.view(2 * hidden, model_dim // TILE, TILE).permute(1, 0, 2).contiguous()
        first_bias = torch.cat([site.w1.bias for site in sites]).to(torch.bfloat16)
        second_layer = torch.stack([_second_layer(site, groups, key_dim, hidden) for site in sites])
        head_bias = torch.stack([site.w2.bias.view(groups, key_dim) for site in sites]).to(torch.bfloat16)
        gate_bias = torch.stack([site.gate.bias for site in sites]).to(torch.bfloat16)
        static_address = torch.stack((static_key, static_query)).to(torch.bfloat16)
        factor = forgetting(forgetting_logit.float().reshape(groups).contiguous(), eps)
        gain = read_gain_parameter.to(torch.bfloat16).float().reshape(groups).contiguous()
        return cls(first_layer, first_bias.contiguous(), second_layer.contiguous(), head_bias.contiguous(),
                   gate_bias.contiguous(), static_address.contiguous(), factor, gain)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        _refuse_training_parameters(state_dict, prefix, set(self._buffers))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class InferenceRule(nn.Module):
    """A block's two edges, by the read that ends them: the attention input and the feed-forward input."""

    def __init__(self, attention: InferenceEdge, feed_forward: InferenceEdge):
        super().__init__()
        self.attention, self.feed_forward = attention, feed_forward

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        _refuse_training_parameters(state_dict, prefix, {"attention", "feed_forward"})
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def _refuse_training_parameters(state_dict, prefix, names):
    foreign = sorted(k for k in state_dict if k.startswith(prefix) and k[len(prefix):].split(".")[0] not in names)
    if foreign:
        raise ValueError(f"{prefix} holds the inference form of the depth mixer, not its training parameters "
                         f"({foreign[0]}, ...): load the checkpoint first, then call prepare_inference")


def _folded_first_layer(site) -> torch.Tensor:
    """``BF16(W * g)``: the first layer with the context gain folded into its input channels, rounded
    as the training forward rounds it."""
    return site.w1.weight.to(torch.bfloat16) * site.context_norm.weight.to(torch.bfloat16).unsqueeze(0)


def _second_layer(site, groups, key_dim, hidden) -> torch.Tensor:
    heads = site.w2.weight.to(torch.bfloat16).view(groups, key_dim, hidden)
    gate = site.gate.weight.to(torch.bfloat16).view(groups, 1, hidden)
    return nn.functional.pad(torch.cat((heads, gate), 1), (0, ROW_PAD))
