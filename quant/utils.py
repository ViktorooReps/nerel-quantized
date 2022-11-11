from typing import TypeVar, Dict, List, Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import pad, one_hot, softmax

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


class FocalLoss(Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.3, gamma: float = 3.0, reduction: str = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction: str = reduction
        self.eps = 1e-6

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # compute softmax over the classes axis
        input_soft = softmax(input, dim=-1) + self.eps

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[-1])

        # compute the actual focal loss
        weight = torch.pow(1. - input_soft, self.gamma)
        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=-1)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}".format(self.reduction))
        return loss
