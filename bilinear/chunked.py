import math
from typing import List, Optional

import torch
from torch import Tensor, bilinear, BoolTensor
from torch.nn import Module, Parameter, init


def split_into_chunks(
        tensor: Tensor,
        chunk_length: int,
        *,
        repeat_chunks: int = 1,
        total_chunks: Optional[int] = None,
        dim: int = 1
) -> Tensor:
    batch_size, length, _ = tensor.shape

    if not length % chunk_length:
        ValueError(f'Length of the input sequence should be a multiple of chunk_length. '
                   f'Got {length} % {chunk_length} = {length % chunk_length}')

    chunks: List[Tensor] = []
    for chunk_i in range(length // chunk_length):
        start_chunk = chunk_length * chunk_i
        end_chunk = start_chunk + chunk_length

        chunk = tensor[:, start_chunk:end_chunk, :].unsqueeze(dim)
        new_chunks = [chunk] * repeat_chunks
        chunks.extend(new_chunks)

    if total_chunks is not None:
        chunks = chunks[:total_chunks]

    return torch.cat(chunks, dim=dim)


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

    def forward(self, from_x: Tensor, to_x: Tensor, chunk_length: int) -> Tensor:
        batch_size, length, _ = from_x.shape

        # chunk input sequences

        total_chunks = 2 * (length // chunk_length) - 1

        from_chunks = split_into_chunks(from_x, chunk_length=chunk_length, repeat_chunks=2, total_chunks=total_chunks)
        to_chunks = split_into_chunks(to_x, chunk_length=chunk_length, repeat_chunks=2, total_chunks=total_chunks)

        # process chunks in batches

        chunked_result = bilinear(
            from_chunks.view(-1, chunk_length, self._in1_features).unsqueeze(-2).repeat(1, 1, chunk_length, 1),
            to_chunks.view(-1, chunk_length, self._in2_features).unsqueeze(-3).repeat(1, chunk_length, 1, 1),
            weight=self._weight,
            bias=self._bias
        )
        return chunked_result.view(batch_size, total_chunks, chunk_length, chunk_length, self._out_features)

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
