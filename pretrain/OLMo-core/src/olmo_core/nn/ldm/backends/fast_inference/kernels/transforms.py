"""The forgetting factor `prepare_inference` computes once per edge: ``max(sigmoid(logit), eps)`` per group,
in the kernels' own FP32 arithmetic (the same factor as the fast backend's ``sites.forgetting_value``)."""

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import Float32, cute

from ..runtime import compile_options, device_key, fake_stream, fake_tensor, on_device, tensor_spec
from .elementwise import sigmoid

THREADS = 128


class _Forgetting:
    def __init__(self, eps: float):
        self.eps = float(eps)

    @cute.jit
    def __call__(self, logits: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
        self.kernel(logits, output).launch(
            grid=(cute.ceil_div(logits.shape[0], THREADS), 1, 1), block=(THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, logits: cute.Tensor, output: cute.Tensor):
        group = cute.arch.block_idx()[0] * THREADS + cute.arch.thread_idx()[0]
        if group < logits.shape[0]:
            output[group] = cutlass.max(sigmoid(logits[group]), Float32(self.eps))


@lru_cache(maxsize=8)
def _program(eps, metadata, capability, device_index):
    del device_index
    return cute.compile(_Forgetting(eps), *(fake_tensor(item) for item in metadata), fake_stream(),
                        options=compile_options(capability))


def forgetting(logits: torch.Tensor, eps: float) -> torch.Tensor:
    """The FP32 forgetting factor of every group from its FP32 logit."""
    if not (logits.is_cuda and logits.dtype == torch.float32 and logits.ndim == 1 and logits.is_contiguous()):
        raise ValueError("a contiguous CUDA FP32 vector of logits is required")
    output = torch.empty_like(logits)
    metadata = tuple(tensor_spec(t) for t in (logits, output))
    with on_device(logits.device):
        _program(float(eps), metadata, *device_key(logits.device))(logits, output)
    return output
