from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


_BLOCKED_LICENSE_TOKENS = (
    "unknown",
    "non-commercial",
    "noncommercial",
    "research-only",
    "research only",
    "educational-only",
)


@dataclass(frozen=True)
class DatasetSource:
    name: str
    root: Path
    samples_index: Path
    owner: str
    license_id: str
    commercial_use_approved: bool
    annotation_use_approved: bool
    consent_basis: str
    approved_by: str
    approved_at: str
    notes: str

    @classmethod
    def from_payload(cls, payload: dict) -> "DatasetSource":
        root = Path(payload["root"]).expanduser()
        return cls(
            name=str(payload["name"]),
            root=root,
            samples_index=root / str(payload["samples_index"]),
            owner=str(payload["owner"]),
            license_id=str(payload["license_id"]),
            commercial_use_approved=bool(payload["commercial_use_approved"]),
            annotation_use_approved=bool(payload["annotation_use_approved"]),
            consent_basis=str(payload["consent_basis"]),
            approved_by=str(payload["approved_by"]),
            approved_at=str(payload["approved_at"]),
            notes=str(payload.get("notes", "")),
        )

    def validate_commercial_use(self) -> list[str]:
        errors: list[str] = []
        license_text = self.license_id.lower()
        if any(token in license_text for token in _BLOCKED_LICENSE_TOKENS):
            errors.append(f"{self.name}: blocked or unknown license_id={self.license_id!r}")
        if not self.commercial_use_approved:
            errors.append(f"{self.name}: commercial_use_approved must be true")
        if not self.annotation_use_approved:
            errors.append(f"{self.name}: annotation_use_approved must be true")
        for field_name, value in (
            ("owner", self.owner),
            ("consent_basis", self.consent_basis),
            ("approved_by", self.approved_by),
            ("approved_at", self.approved_at),
        ):
            if not value.strip():
                errors.append(f"{self.name}: {field_name} is required")
        if not self.samples_index.is_file():
            errors.append(f"{self.name}: missing samples index {self.samples_index}")
        return errors


def load_sources(path: Path) -> list[DatasetSource]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest schema_version")
    sources = [DatasetSource.from_payload(item) for item in payload.get("datasets", [])]
    if not sources:
        raise ValueError("Dataset manifest must contain at least one dataset")
    return sources

