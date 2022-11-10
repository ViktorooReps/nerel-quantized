"""
SOURCE: https://github.com/huggingface/transformers/blob/main/examples/research_projects/bertology/run_bertology.py
"""
import logging

import torch
from torch import Tensor, BoolTensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from quant.datamodel import BatchedExamples

logger = logging.getLogger(__name__)


def compute_heads_importance(model, eval_dataloader: DataLoader, head_mask: Tensor = None) -> Tensor:
    """This method shows how to compute:
    - head attention entropy
    - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    n_layers, n_heads = model._encoder.config.num_hidden_layers, model._encoder.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(model.device)

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(model.device)

    head_mask.requires_grad_(requires_grad=True)

    tot_tokens = 0.0

    for step, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        examples: BatchedExamples = inputs['examples']

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        model._encoder.config.output_attentions = True
        loss, _, all_attentions = model(**inputs, encoder_head_mask=head_mask, return_attention_scores=True)
        loss.backward()  # Backpropagate to populate the gradients in the head mask

        head_importance += head_mask.grad.abs().detach()

        tot_tokens += examples.padding_mask.float().detach().sum().data

    # Normalize
    head_importance /= tot_tokens
    # Layerwise importance normalization

    exponent = 2
    norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
    head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    head_importance = (head_importance - head_importance.min()) / (head_importance.max() - head_importance.min())

    return head_importance


def mask_heads(model, eval_dataloader: DataLoader, prune_fraction: float = 0.1, num_iter: int = 5) -> BoolTensor:
    """This method shows how to mask head (set some heads to zero), to test the effect on the network,
    based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    head_importance = compute_heads_importance(model, eval_dataloader)

    new_head_mask = torch.ones_like(head_importance)
    head_mask = new_head_mask.clone()

    total_num_to_mask = max(1, int(new_head_mask.numel() * prune_fraction))
    left_to_mask = total_num_to_mask

    for masking_iter in range(num_iter):
        if left_to_mask <= 0:
            continue

        num_to_mask = max(1, total_num_to_mask // num_iter)
        left_to_mask -= num_to_mask

        head_mask = new_head_mask.clone()  # save current head mask
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        # mask heads
        current_heads_to_mask = current_heads_to_mask[:num_to_mask]
        logger.info("Heads to mask: %s", str(current_heads_to_mask.tolist()))
        new_head_mask = new_head_mask.view(-1)
        new_head_mask.data[current_heads_to_mask] = 0.0
        new_head_mask = new_head_mask.view_as(head_mask)
        new_head_mask = new_head_mask.clone().detach()

        # Compute metric and head importance again
        head_importance = compute_heads_importance(model, eval_dataloader, head_mask=new_head_mask)

    return head_mask


def prune_heads(model, head_mask: BoolTensor):
    """This method shows how to prune head (remove heads weights) based on
    the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    heads_to_prune = dict(
        (layer, (1 - head_mask[layer].long()).nonzero().squeeze().tolist()) for layer in range(len(head_mask))
    )

    model._encoder.prune_heads(heads_to_prune)  # TODO
