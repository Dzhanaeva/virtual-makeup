from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from lipseg.splits import subject_split

@dataclass(frozen=True)
class PreparedSample:
    image: Path
    mask: Path
    subject_id: str
    consent_record_id: str


def read_prepared_index(path: Path) -> list[PreparedSample]:
    samples: list[PreparedSample] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            samples.append(
                PreparedSample(
                    image=path.parent / payload["image"],
                    mask=path.parent / payload["mask"],
                    subject_id=payload["subject_id"],
                    consent_record_id=payload["consent_record_id"],
                )
            )
        except KeyError as error:
            raise ValueError(f"{path}:{line_number}: missing {error.args[0]}") from error
    return samples


class LipROIDataset(Dataset):
    def __init__(self, index_path: Path, augment: bool) -> None:
        self.samples = read_prepared_index(index_path)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image).convert("RGB")
        mask = Image.open(sample.mask).convert("L")
        if self.augment:
            image, mask = self._augment(image, mask)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.int64)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array)
        return image_tensor, mask_tensor

    @staticmethod
    def _augment(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.65:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.58, 1.28))
        if random.random() < 0.55:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.72, 1.30))
        if random.random() < 0.40:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.72, 1.24))
        if random.random() < 0.30:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.35)))
        return image, mask
