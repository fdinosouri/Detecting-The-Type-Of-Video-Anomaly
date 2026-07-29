"""Losses and class-weighting utilities for long-tailed classification.

UCF-Crime is extremely imbalanced (Normal dominates; several anomaly
classes have only a handful of test videos). Plain cross-entropy lets the
model collapse onto the head classes, which is exactly what a low macro-F1
with several 0.00 rows looks like. Two standard fixes are combined here:

* class-balanced weights (Cui et al., CVPR 2019, "effective number of
  samples") so rare classes contribute comparable gradient mass, and
* focal loss (Lin et al., ICCV 2017) so easy Normal clips stop dominating
  the loss once they are confidently classified.
"""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_counts(labels, num_classes: int) -> torch.Tensor:
    counter = Counter(labels)
    return torch.tensor(
        [max(counter.get(c, 0), 1) for c in range(num_classes)], dtype=torch.float32
    )


def class_balanced_weights(counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / effective_num
    return weights / weights.sum() * len(counts)


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        if self.label_smoothing > 0:
            true_dist = torch.full_like(
                log_probs, self.label_smoothing / (num_classes - 1)
            )
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            true_dist = F.one_hot(target, num_classes).to(log_probs.dtype)

        focal_factor = (1.0 - probs).pow(self.gamma)
        loss = -(true_dist * focal_factor * log_probs)
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device).unsqueeze(0)
        return loss.sum(dim=-1).mean()


def build_criterion(cfg: dict, train_labels, num_classes: int) -> nn.Module:
    loss_cfg = cfg.get("loss", {})
    weights = None
    if loss_cfg.get("class_balanced", True):
        counts = class_counts(train_labels, num_classes)
        weights = class_balanced_weights(counts, beta=loss_cfg.get("cb_beta", 0.9999))
    gamma = loss_cfg.get("focal_gamma", 2.0)
    label_smoothing = loss_cfg.get("label_smoothing", 0.1)
    if gamma > 0:
        return FocalLoss(weight=weights, gamma=gamma, label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
