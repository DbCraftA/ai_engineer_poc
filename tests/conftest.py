"""Shared pytest helpers for the inference-lab roadmap."""

from importlib.util import find_spec

import pytest
import torch


def pytest_runtest_setup(item):
    """Skip hardware-specific tests when the current environment cannot run them."""
    if "cuda" in item.keywords and not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment.")

    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("GPU tests require a CUDA-capable environment for now.")

    if "triton" in item.keywords:
        if find_spec("triton") is None:
            pytest.skip("Triton is not installed in this environment.")
        if not torch.cuda.is_available():
            pytest.skip("Triton tests require CUDA in this roadmap.")

    if "distributed" in item.keywords and find_spec("torch.distributed") is None:
        pytest.skip("torch.distributed is not available in this environment.")
