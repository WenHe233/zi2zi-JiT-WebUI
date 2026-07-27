from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

import torch


Function = TypeVar("Function", bound=Callable)


def torch_compile_available() -> bool:
    mode = os.environ.get("ZI2ZI_TORCH_COMPILE", "auto").strip().lower()
    if mode in {"0", "false", "no", "off"}:
        return False
    if mode in {"1", "true", "yes", "on"}:
        return True
    try:
        from torch.utils._triton import has_triton

        return bool(has_triton())
    except (ImportError, RuntimeError):
        return False


def compile_if_available(function: Function) -> Function:
    """Use torch.compile only when its Triton backend is usable."""
    if not torch_compile_available():
        return function
    return torch.compile(function)
