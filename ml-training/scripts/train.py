#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from lipseg.config import load_config
from lipseg.data import LipROIDataset
from lipseg.losses import CrossEntropyDiceLoss
from lipseg.metrics import intersection_and_union, mean_foreground_iou
from lipseg.model import LipSegmentationNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scratch-only lip ROI segmentation model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=TRAINING_ROOT / "configs" / "lip_roi.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def data_loader(data_root: Path, split: str, batch_size: int, workers: int, augment: bool) -> DataLoader:
    dataset = LipROIDataset(data_root / split / "index.jsonl", augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=workers,
        pin_memory=False,
    )


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.shape[0]
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_count: int) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    intersection = torch.zeros(class_count, dtype=torch.float64)
    union = torch.zeros(class_count, dtype=torch.float64)
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        total_loss += float(criterion(logits, masks).item()) * images.shape[0]
        batch_intersection, batch_union = intersection_and_union(logits.cpu(), masks.cpu(), class_count)
        intersection += batch_intersection
        union += batch_union
    return total_loss / len(loader.dataset), mean_foreground_iou(intersection, union)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    train_loader = data_loader(args.data, "train", args.batch_size, args.workers, augment=True)
    validation_loader = data_loader(args.data, "validation", args.batch_size, args.workers, augment=False)
    model = LipSegmentationNet(class_count=len(config.classes)).to(device)
    criterion = CrossEntropyDiceLoss(class_count=len(config.classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_iou = -1.0
    print(f"training device={device} train={len(train_loader.dataset)} validation={len(validation_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        validation_loss, validation_iou = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            len(config.classes),
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"validation_loss={validation_loss:.5f} foreground_iou={validation_iou:.5f}"
        )
        if validation_iou > best_iou:
            best_iou = validation_iou
            checkpoint = {
                "state_dict": model.state_dict(),
                "width": config.width,
                "height": config.height,
                "classes": config.classes,
                "foreground_iou": best_iou,
            }
            torch.save(checkpoint, args.output / "lipseg-best.pt")
            (args.output / "metrics.json").write_text(
                json.dumps({"best_foreground_iou": best_iou, "epoch": epoch}, indent=2) + "\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
