"""Scalar device functions shared by the fast-inference kernels."""

from cutlass import Float32, cute


@cute.jit
def sigmoid(x: Float32):
    """``1 / (1 + exp(-x))`` in FP32: the expression every kernel of this backend uses."""
    return Float32(1.0) / (Float32(1.0) + cute.math.exp(-x))
