from __future__ import annotations

import logging
import re
import warnings
from typing import Optional, Tuple

import torch
from torch import nn


_LOGGER = logging.getLogger(__name__)


def _parse_torch_version(version: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"(\d+)\.(\d+)", version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def maybe_compile_model(model: nn.Module, enabled: bool, quiet: bool = False) -> nn.Module:
    """Apply torch.compile() to the model when available and requested."""
    if not enabled:
        return model

    if not torch.cuda.is_available():
        warnings.warn("torch.compile() skipped: CUDA not available.")
        return model

    torch_version = _parse_torch_version(torch.__version__)
    if torch_version is None:
        warnings.warn(f"Could not parse PyTorch version '{torch.__version__}'. Skipping compilation.")
        return model

    if torch_version < (2, 0):
        warnings.warn(
            f"torch.compile() requires PyTorch 2.0+, found {torch.__version__}. "
            "Skipping compilation."
        )
        return model

    if not hasattr(torch, "compile"):
        warnings.warn(f"torch.compile() not available in PyTorch {torch.__version__}. Skipping compilation.")
        return model

    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
    except Exception as exc:
        warnings.warn(f"torch.compile() failed: {exc}. Continuing without compilation.")
        return model

    if not quiet:
        _LOGGER.info(
            "torch.compile() enabled (mode=reduce-overhead). "
            "First batch may be slower due to JIT compilation."
        )

    return compiled_model
