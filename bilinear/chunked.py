import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor, bilinear, BoolTensor, LongTensor
from torch.nn import Module, Parameter, init, Bilinear


def split_into_chunks(
        tensor: Tensor,
        chunk_length: int,
        *,
        repeat_chunks: int = 1,
        drop_first: bool = False,
        drop_last: bool = False,
        idx_style: str = 'col',
        dim: int = 1
) -> Tuple[Tensor, LongTensor]:
    batch_size, length, _ = tensor.shape

    if not length % chunk_length:
        ValueError(f'Length of the input sequence should be a multiple of chunk_length. '
                   f'Got {length} % {chunk_length} = {length % chunk_length}')

    chunks: List[Tensor] = []
    indices: List[LongTensor] = []
    for chunk_i in range(length // chunk_length):
        start_chunk = chunk_length * chunk_i
        end_chunk = start_chunk + chunk_length

        chunk = tensor[:, start_chunk:end_chunk, :].unsqueeze(dim)
        new_chunks = [chunk] * repeat_chunks
        chunks.extend(new_chunks)

        if idx_style == 'col':
            idx = torch.arange(start_chunk, end_chunk, dtype=torch.long, device=tensor.device).repeat(chunk_length)
        elif idx_style == 'row':
            idx = torch.arange(start_chunk, end_chunk, dtype=torch.long, device=tensor.device).repeat_interleave(chunk_length)
        else:
            raise ValueError

        indices.extend([idx] * repeat_chunks)

    if drop_first:
        chunks = chunks[:-1]
        indices = indices[:-1]

    if drop_last:
        chunks = chunks[1:]
        indices = indices[1:]

    return torch.cat(chunks, dim=dim), torch.cat(indices).long()


class ChunkedBilinear(Module):
    """Applies bilinear transformation to quasi-diagonal elements+1 of the output matrix:

   >|------|< chunk_length
    |------|------|-------------|
    |  1   |  2   |             |
    |------|------|------|      |
    |      |  3   |  4   |      |
    |      |------|------|      |
    |        . . . . . . . . . .|
    |             |------|------|
    |             | n-2  | n-1  |
    |             |------|------|
    |                    |  n   |
    |--------------------|------|

    Returns chunks in the order illustrated above.
    """
    __constants__ = ['_in1_features', '_in2_features', '_out_features']
    _in1_features: int
    _in2_features: int
    _out_features: int

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self._in1_features = in1_features
        self._in2_features = in2_features
        self._out_features = out_features

        self._weight = Parameter(torch.empty(out_features, in1_features, in2_features))
        self._bias: Optional[Parameter] = None
        if bias:
            self.bias = Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self._weight.size(1))
        init.uniform_(self._weight, -bound, bound)
        if self._bias is not None:
            init.uniform_(self._bias, -bound, bound)

    def forward(self, from_x: Tensor, to_x: Tensor, chunk_length: int) -> Tuple[Tensor, Tuple[LongTensor, LongTensor]]:
        batch_size, length, _ = from_x.shape

        # chunk input sequences

        total_chunks = 2 * (length // chunk_length) - 1

        from_chunks, row_idx = split_into_chunks(from_x, chunk_length=chunk_length, repeat_chunks=2, drop_first=True, idx_style='row')
        to_chunks, col_idx = split_into_chunks(to_x, chunk_length=chunk_length, repeat_chunks=2, drop_last=True)

        # process chunks in batches

        chunked_result = bilinear(
            from_chunks.view(-1, chunk_length, self._in1_features).unsqueeze(-2).repeat(1, 1, chunk_length, 1),
            to_chunks.view(-1, chunk_length, self._in2_features).unsqueeze(-3).repeat(1, chunk_length, 1, 1),
            weight=self._weight,
            bias=self._bias
        )
        return chunked_result.view(batch_size, total_chunks, chunk_length, chunk_length, self._out_features), (row_idx, col_idx)

    @staticmethod
    def output_mask(length: int, chunk_length: int) -> BoolTensor:
        if not length % chunk_length:
            ValueError(f'Argument length should be a multiple of chunk_length. Got {length} % {chunk_length} = {length % chunk_length}')

        def block(height: int, width:  int, *, value: bool = True) -> BoolTensor:
            return torch.full([height, width], fill_value=value, dtype=torch.bool).bool()

        num_diag_blocks = length // chunk_length
        if num_diag_blocks == 1:
            return block(length, length)

        start_block = block(chunk_length, chunk_length * 2)
        diag = torch.block_diag(*[block(chunk_length, chunk_length) for _ in range(num_diag_blocks)])

        if num_diag_blocks == 2:
            return start_block | diag

        end_block = block(chunk_length * 2, chunk_length)

        diag_p1_blocks = [start_block] + [block(chunk_length, chunk_length) for _ in range(num_diag_blocks - 3)] + [end_block]
        return torch.block_diag(*diag_p1_blocks) | diag


if __name__ == '__main__':
    batch_size = 1
    sequence_length = 10
    chunk_length = 2

    chunked_bili = ChunkedBilinear(100, 100, 5)
    bili = Bilinear(100, 100, 5)
    bili.weight = chunked_bili._weight
    bili.bias = chunked_bili._bias

    inp = torch.randn((batch_size, sequence_length, 100))
    outp = torch.zeros((batch_size, sequence_length, sequence_length, 5))

    res = bili(inp.unsqueeze(-2).repeat(1, 1, sequence_length, 1), inp.unsqueeze(-3).repeat(1, sequence_length, 1, 1))
    chunked_res, (row_idx, col_idx) = chunked_bili(inp, inp, 2)

    outp[:, row_idx, col_idx, :] = chunked_res.view(batch_size, -1, 5)

    positions = torch.arange(10)
    start_positions = positions.unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, sequence_length)
    end_positions = positions.unsqueeze(-2).unsqueeze(0).repeat(batch_size, sequence_length, 1)

    assert torch.allclose(res[:, row_idx, col_idx], chunked_res.view(-1, 5))
