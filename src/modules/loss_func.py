import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict
from torch.nn import Module


class ATLoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, logits, labels):
        """
        logits & labels: [batch_size, num_class]
        ========================================
        Reference: ATLOP code
        """
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2 * self.beta
        loss = loss.mean()
        return loss


class SCELoss(nn.Module):
    def __init__(self, beta, s0=0):
        super().__init__()
        self.beta = beta
        self.s0 = s0

    def forward(self, logits, labels):
        spos = torch.clone(logits).fill_(-self.s0)
        sneg = torch.clone(logits).fill_(self.s0)
        # 多标签分类的交叉熵
        logits = (1 - 2 * labels) * logits
        y_pred_pos = logits - (1 - labels) * 1e30
        y_pred_neg = logits - labels * 1e30
        zeros = torch.zeros_like(logits[..., :1])
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        pos_loss = torch.log(torch.sum(torch.exp(y_pred_pos)) + torch.exp(spos))
        # pos_loss = torch.log1p(torch.sum(torch.exp(y_pred_pos)))
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        neg_loss = torch.log(torch.sum(torch.exp(y_pred_neg)) + torch.exp(sneg))
        # neg_loss = torch.log1p(torch.sum(torch.exp(y_pred_neg)))
        loss = pos_loss + neg_loss * self.beta
        loss = loss.mean()
        return loss


class categorical_crossentropy(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, logits, labels):
        logits = (1 - 2 * labels) * logits
        y_pred_pos = logits - (1 - labels) * 1e30
        y_pred_neg = logits - labels * 1e30
        zeros = torch.zeros_like(logits[..., :1])
        y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
        neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
        loss = pos_loss + neg_loss * self.beta
        loss = loss.mean()
        return loss


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def masked_log_softmax(vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1) -> torch.Tensor:
    """
    `torch.nn.functional.log_softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a log_softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular log_softmax.
    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not `nan`.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you `nans`.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + (mask + tiny_value_of_dtype(vector.dtype)).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def contrastive_loss(
        scores: torch.FloatTensor,
        positions: Union[List[int], Tuple[List[int], List[int]]],
        mask: torch.BoolTensor,
        prob_mask: torch.BoolTensor = None,
) -> torch.FloatTensor:
    seq_len = scores.shape[0]
    if len(scores.shape) == 2:
        scores = scores.view(-1)
        mask = mask.view(-1)
        log_probs = masked_log_softmax(scores, mask)
        log_probs = log_probs.view(seq_len, seq_len)
        start_positions, end_positions = positions
        # log_probs = log_probs[start_positions, end_positions]
    elif len(scores.shape) == 3:
        type_len = scores.shape[2]
        scores = scores.view(-1, type_len)
        mask = mask.view(-1, type_len)
        log_probs = masked_log_softmax(scores, mask)
        log_probs = log_probs.view(seq_len, seq_len, type_len)
        start_positions, end_positions = positions
        batch_indices = list(range(type_len))
        # log_probs = log_probs[start_positions, end_positions, batch_indices]
    return - log_probs.mean()
