from __future__ import annotations

import torch


def intersection_and_union(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = logits.argmax(dim=1)
    intersection = torch.zeros(class_count, dtype=torch.float64)
    union = torch.zeros(class_count, dtype=torch.float64)
    for class_id in range(class_count):
        predicted = prediction == class_id
        expected = target == class_id
        intersection[class_id] = torch.logical_and(predicted, expected).sum().item()
        union[class_id] = torch.logical_or(predicted, expected).sum().item()
    return intersection, union


def mean_foreground_iou(intersection: torch.Tensor, union: torch.Tensor) -> float:
    valid_union = union[1:]
    valid_intersection = intersection[1:]
    present = valid_union > 0
    if not present.any():
        return 0.0
    return float((valid_intersection[present] / valid_union[present]).mean().item())
