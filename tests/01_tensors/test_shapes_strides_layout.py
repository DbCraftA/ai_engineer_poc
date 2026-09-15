import torch


def test_tensor_shape_describes_logical_dimensions():
    # Concept: a shape tells us how many elements exist along each logical axis.
    # Here: B = batch size, T = sequence length, C = hidden dimension.
    x = torch.zeros(2, 3, 4)

    assert x.shape == torch.Size([2, 3, 4])
    assert x.numel() == 2 * 3 * 4


def test_stride_describes_how_indices_move_through_storage():
    # In a contiguous [B, T, C] tensor, moving by one batch jumps T*C values,
    # moving by one token jumps C values, and moving by one channel jumps 1 value.
    x = torch.zeros(2, 3, 4)

    assert x.stride() == (12, 4, 1)


def test_transpose_changes_view_strides_without_copying_storage():
    # Transpose does not need to move data immediately. It creates another view
    # where the same storage is interpreted with different strides.
    x = torch.arange(2 * 3).reshape(2, 3)
    y = x.transpose(0, 1)

    assert y.shape == torch.Size([3, 2])
    assert y.stride() == (1, 3)
    assert y.storage().data_ptr() == x.storage().data_ptr()


def test_contiguous_materializes_a_layout_with_standard_strides():
    # Some kernels need contiguous memory. Calling contiguous() copies data only
    # when the current layout is non-contiguous.
    x = torch.arange(2 * 3).reshape(2, 3)
    transposed = x.transpose(0, 1)
    contiguous = transposed.contiguous()

    assert not transposed.is_contiguous()
    assert contiguous.is_contiguous()
    assert contiguous.shape == transposed.shape
    assert contiguous.stride() == (2, 1)


def test_views_can_share_storage_but_interpret_it_differently():
    # A view can share the same underlying memory while exposing a different
    # logical shape. This is central for reshape/view operations in attention.
    x = torch.arange(12)
    y = x.view(3, 4)

    assert y.shape == torch.Size([3, 4])
    assert y.storage().data_ptr() == x.storage().data_ptr()
    assert y[2, 3].item() == x[11].item()

