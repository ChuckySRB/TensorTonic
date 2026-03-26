import numpy as np


def _conv2d_same(x: np.ndarray, kernels: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    2D convolution with stride=1 and "same" padding for NHWC inputs.
    """
    batch_size, height, width, in_channels = x.shape
    kernel_h, kernel_w, _, out_channels = kernels.shape

    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    x_padded = np.pad(
        x,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode="constant",
    )

    out = np.zeros((batch_size, height, width, out_channels), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            patch = x_padded[:, i : i + kernel_h, j : j + kernel_w, :]
            # Tensordot contracts over (kh, kw, in_channels) and keeps batch/out channels.
            out[:, i, j, :] = np.tensordot(
                patch,
                kernels,
                axes=([1, 2, 3], [0, 1, 2]),
            ) + bias

    return out


def _max_pool2d_2x2_stride2(x: np.ndarray) -> np.ndarray:
    """
    2x2 max-pooling with stride 2 for NHWC inputs.
    """
    batch_size, height, width, channels = x.shape
    out_h = height // 2
    out_w = width // 2

    pooled = np.zeros((batch_size, out_h, out_w, channels), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            window = x[:, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2, :]
            pooled[:, i, j, :] = np.max(window, axis=(1, 2))

    return pooled


def vgg_conv_block(
    x: np.ndarray,
    num_convs: int,
    out_channels: int,
    apply_pool: bool = False,
) -> np.ndarray:
    """
    Implement a VGG-style convolutional block.

    The block performs:
    1) `num_convs` convolution layers (3x3, stride 1, same padding)
    2) ReLU after each convolution
    3) Optional 2x2 max-pooling with stride 2 (`apply_pool=True`)

    Expected input shape: (batch, height, width, channels) (NHWC)
    Default output shape (without pooling): (batch, height, width, out_channels)
    """
    if x.ndim != 4:
        raise ValueError("Input x must be a 4D tensor with shape (batch, height, width, channels)")
    if num_convs < 1:
        raise ValueError("num_convs must be at least 1")
    if out_channels < 1:
        raise ValueError("out_channels must be at least 1")

    x = x.astype(np.float32, copy=False)
    input_dim = x.shape[-1]

    for _ in range(num_convs):
        # He initialization is common for ReLU-based convolution stacks.
        kernels = np.random.randn(3, 3, input_dim, out_channels).astype(np.float32)
        kernels *= np.sqrt(2.0 / (3 * 3 * input_dim))
        bias = np.zeros((out_channels,), dtype=np.float32)

        x = _conv2d_same(x, kernels, bias)
        x = np.maximum(x, 0.0)  # ReLU
        input_dim = out_channels

    if apply_pool:
        x = _max_pool2d_2x2_stride2(x)
    return x