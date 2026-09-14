import pytest
import torch


@pytest.mark.gpu
@pytest.mark.cuda
def test_cuda_tensor_allocation_is_skipped_when_cuda_is_unavailable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment.")

    x = torch.ones(4, device="cuda")

    assert x.device.type == "cuda"

