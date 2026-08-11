from __future__ import annotations

import torch
from torch import nn


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.class_count = class_count
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.cross_entropy(logits, target)
        probabilities = torch.softmax(logits, dim=1)
        one_hot = nn.functional.one_hot(target, num_classes=self.class_count)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
        denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2 * intersection + 1) / (denominator + 1)
        return ce_loss + (1 - dice.mean())

