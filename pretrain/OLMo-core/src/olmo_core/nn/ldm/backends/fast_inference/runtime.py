"""CuTe operand specs with a runtime token axis, and how the backend's kernels are compiled.

Every kernel is compiled with the TVM-FFI launcher: it is called with the PyTorch tensors
themselves (pointer, shape and strides are read in C, the token extent is bound at run time) and
launches on PyTorch's current stream. No per-call operand conversion happens in Python.
"""

import contextlib
from functools import lru_cache

import torch

# Every kernel operand starts on a 16-byte boundary.
ALIGNMENT = 16


def spec(dtype, shape):
    """A hashable compile-time description of a contiguous operand of `shape`, where None marks the runtime
    token extent."""
    return dtype, tuple(shape), ALIGNMENT


def tensor_spec(tensor, token_axis=None):
    """The `spec` of a contiguous operand whose `token_axis` stays a runtime extent."""
    check_operands(tensor)
    shape = list(tensor.shape)
    if token_axis is not None:
        shape[token_axis] = None
    return spec(tensor.dtype, shape)


def check_operands(*tensors):
    """Operands from outside the backend must be contiguous and 16-byte aligned: the kernels move them with
    bulk copies and 16-byte vector accesses."""
    for tensor in tensors:
        if not tensor.is_contiguous() or tensor.data_ptr() % ALIGNMENT:
            raise ValueError("inference kernel operands must be contiguous and 16-byte aligned")


@lru_cache(maxsize=None)
def device_key(device: torch.device) -> tuple[tuple[int, int], int]:
    """The compute capability and index of a CUDA device, the device part of every compile key."""
    return torch.cuda.get_device_capability(device.index), device.index


def on_device(device: torch.device):
    """The context that makes `device` current, when it is not already, for a kernel launch."""
    if device.index == torch.cuda.current_device():
        return contextlib.nullcontext()
    return torch.cuda.device(device)


def fake_tensor(spec):
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import make_fake_compact_tensor

    dtype, shape, alignment = spec
    types = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}
    shape = tuple(cute.sym_int32() if s is None else s for s in shape)
    return make_fake_compact_tensor(types[dtype], shape,
                                    stride_order=tuple(reversed(range(len(shape)))),
                                    assumed_align=alignment)


def fake_stream():
    """The compile-time stream argument, bound to PyTorch's current stream at every call."""
    from cutlass.cute.runtime import make_fake_stream

    return make_fake_stream(use_tvm_ffi_env_stream=True)


def compile_options(capability: tuple[int, int]) -> str:
    """`cute.compile` options for a device of this compute capability (architecture-specific
    features on SM90 and SM100) with the TVM-FFI launcher."""
    major, minor = capability
    arch = {(9, 0): "sm_90a", (10, 0): "sm_100a"}.get(capability, f"sm_{major}{minor}")
    return f"--gpu-arch={arch} --enable-tvm-ffi"
