import torch



def bytes_per_element(dtype: torch.dtype) -> int:
    return dtype.itemsize


def roundtrip(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    pass

def max_absolute_error(tensor: torch.Tensor, dtype: torch.dtype) -> float:
    pass