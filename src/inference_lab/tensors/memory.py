from .dtypes import bytes_per_element
import torch

def tensor_memory_bytes(tensor: torch.Tensor):
    return tensor.numel() * bytes_per_element(tensor.dtype)