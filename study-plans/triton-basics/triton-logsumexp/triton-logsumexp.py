import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis = 0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x_row_ptr = x_ptr + row_idx * x_row_stride + cols
    row = tl.load(x_row_ptr, mask = mask, other = -float('inf'))

    row_max = tl.max(row, axis = 0)
    row_sum = tl.sum(tl.exp(row - row_max), axis = 0)
    lse = row_max + tl.log(row_sum)

    tl.store(out_ptr + row_idx, lse)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )