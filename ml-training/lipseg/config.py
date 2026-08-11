from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LipROIConfig:
    input_width: int
    input_height: int
    class_count: int
    class_names: tuple[str, ...]
    roi_margin_x: float
    roi_margin_y: float
    split_seed: str
    splits: dict[str, float]

    @classmethod
    def load(cls, path: Path) -> "LipROIConfig":
        payload = json.loads(path.read_text())
        config = cls(
            input_width=int(payload["input_width"]),
            input_height=int(payload["input_height"]),
            class_count=int(payload["class_count"]),
            class_names=tuple(payload["class_names"]),
            roi_margin_x=float(payload["roi_margin_x"]),
            roi_margin_y=float(payload["roi_margin_y"]),
            split_seed=str(payload["split_seed"]),
            splits={name: float(value) for name, value in payload["splits"].items()},
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.input_width <= 0 or self.input_height <= 0:
            raise ValueError("ROI dimensions must be positive")
        if self.class_count != len(self.class_names):
            raise ValueError("class_count must match class_names")
        if set(self.splits) != {"train", "validation", "test"}:
            raise ValueError("splits must contain train, validation, and test")
        if abs(sum(self.splits.values()) - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to one")

    @property
    def width(self) -> int:
        return self.input_width

    @property
    def height(self) -> int:
        return self.input_height

    @property
    def classes(self) -> list[str]:
        return list(self.class_names)

    @property
    def margin_x(self) -> float:
        return self.roi_margin_x

    @property
    def margin_y(self) -> float:
        return self.roi_margin_y


def load_config(path: Path) -> LipROIConfig:
    return LipROIConfig.load(path)
