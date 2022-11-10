from functools import partial
from typing import TypeVar, Dict, List, Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import pad, binary_cross_entropy_with_logits

_K = TypeVar('_K')
_V = TypeVar('_V')


def invert(d: Dict[_K, _V]) -> Dict[_V, _K]:
    return {v: k for k, v in d.items()}


_Tensor = TypeVar('_Tensor', bound=Tensor)


def pad_images(images: List[_Tensor], *, padding_value: Any = 0.0, padding_length: Optional[int] = None) -> _Tensor:
    """Pad images to equal length (maximum height and width)."""
    max_height, max_width = padding_length, padding_length
    if padding_length is None:
        shapes = torch.tensor(list(map(lambda t: t.shape, images)), dtype=torch.long).transpose(0, 1)
        max_height, max_width = shapes[-2].max(), shapes[-1].max()

    ignore_dims = len(images[0].shape) - 2

    image_batch = [
        # The needed padding is the difference between the
        # max width/height and the image's actual width/height.
        pad(img, [*([0, 0] * ignore_dims), 0, max_width - img.shape[-1], 0, max_height - img.shape[-2]], value=padding_value)
        for img in images
    ]
    return torch.stack(image_batch)


def to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


FocalLoss = lambda reduction: partial(sigmoid_focal_loss, reduction=reduction)

