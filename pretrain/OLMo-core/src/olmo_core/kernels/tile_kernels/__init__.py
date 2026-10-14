"""Vendored subset of DeepSeek's TileKernels (https://github.com/deepseek-ai/TileKernels, MIT):
the mHC (manifold-constrained Hyper-Connections) TileLang kernels and modeling layers of the
mHC paper's section 4.3. See VENDORED.md for the upstream commit and the local changes."""
import tilelang  # noqa: F401  (TileLang runtime; needs a CUDA toolkit on PATH)

from . import config, mhc, modeling, torch  # noqa: E402
from .config import get_device_num_sms, get_num_sms, set_num_sms  # noqa: E402
