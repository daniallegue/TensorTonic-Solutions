import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis = 0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x_row_ptr = x_ptr + row_idx * x_row_stride + cols
    row = tl.load(x_row_ptr, mask = mask, other = -float('inf'))

    row_max = tl.max(row, axis = 0)
    row = row - row_max
    row = tl.exp(row)
    row_sum = tl.sum(row, axis = 0)
    row = row / row_sum
    
    out_row_ptr = out_ptr + row_idx * out_row_stride + cols
    tl.store(out_row_ptr, row, mask = mask)

    
def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )