"""JSON-ready experiment records. All timings are seconds and memory values are bytes."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FrameEvidence:
    target_timestamp_s: float
    timestamp_s: float
    presentation_timestamp_s: float
    pts: int
    time_base: str
    frame_index: int
    sha256_rgb: str
    width: int
    height: int
    image_png_base64: str


@dataclass
class Trial:
    phase: str
    index: int
    status: str = "ok"
    answer: str | None = None
    error: str | None = None
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    unavailable_reasons: dict[str, str] = field(default_factory=dict)
    processing: dict[str, Any] = field(default_factory=dict)
    evidence_hashes: list[str] = field(default_factory=list)


@dataclass
class Configuration:
    requested_frames: int
    selected_frames: int = 0
    status: str = "ok"
    error: str | None = None
    evidence: list[FrameEvidence] = field(default_factory=list)
    trials: list[Trial] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)


@dataclass
class Comparison:
    run_id: str
    created_at_utc: str
    backend: str
    video: dict[str, Any]
    question: str
    settings: dict[str, Any]
    environment: dict[str, Any]
    model: dict[str, Any]
    configurations: list[Configuration] = field(default_factory=list)
    fixture: dict[str, Any] | None = None
    schema_version: str = "1.0.0"

    @property
    def complete(self) -> bool:
        return bool(self.configurations) and all(c.status == "ok" for c in self.configurations)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def export_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
        return path

    def export_html(self, path: str | Path) -> Path:
        from vidfit.evaluation.reports import render_html

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(self), encoding="utf-8")
        return path
