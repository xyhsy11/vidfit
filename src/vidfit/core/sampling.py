"""Immutable selection requests shared by the API, CLI, previews and inference."""

import math
from dataclasses import dataclass
from numbers import Real


def midpoint_targets(duration_s: float, frame_budget: int) -> list[float]:
    if not isinstance(frame_budget, int) or isinstance(frame_budget, bool) or frame_budget < 1:
        raise ValueError("frame budget must be a positive integer")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("video duration must be finite and positive")
    return [(i + 0.5) * duration_s / frame_budget for i in range(frame_budget)]


@dataclass(frozen=True)
class SelectionSpec:
    method: str = "uniform"
    frame_budget: int | None = 4
    timestamps_s: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "timestamps_s", tuple(self.timestamps_s))
        if self.method == "uniform":
            midpoint_targets(1, self.frame_budget)
            if self.timestamps_s:
                raise ValueError("uniform selection cannot contain manual timestamps")
        elif self.method == "manual":
            if self.frame_budget is not None or not self.timestamps_s:
                raise ValueError("manual selection needs timestamps and frame_budget=None")
            if any(
                isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t)
                for t in self.timestamps_s
            ):
                raise ValueError("manual timestamps must be finite numbers in seconds")
            object.__setattr__(self, "timestamps_s", tuple(float(t) for t in self.timestamps_s))
            if len(set(self.timestamps_s)) != len(self.timestamps_s):
                raise ValueError(
                    "manual timestamps must resolve to distinct frames; duplicates given"
                )
        else:
            raise ValueError("selection method must be uniform or manual")

    @classmethod
    def uniform(cls, frame_budget=4):
        return cls("uniform", frame_budget)

    @classmethod
    def manual(cls, timestamps_s):
        return cls("manual", None, tuple(timestamps_s))

    @property
    def count(self):
        return self.frame_budget if self.method == "uniform" else len(self.timestamps_s)

    def targets(self, duration_s):
        midpoint_targets(duration_s, 1)  # also validate duration for manual selection
        if self.method == "uniform":
            return midpoint_targets(duration_s, self.frame_budget)
        if any(t < 0 or t >= duration_s for t in self.timestamps_s):
            raise ValueError(f"manual timestamps must be within [0, {duration_s:g}) seconds")
        return sorted(float(t) for t in self.timestamps_s)

    def to_dict(self):
        return {
            "method": self.method,
            "frame_budget": self.count,
            "requested_timestamps_s": list(self.timestamps_s) if self.method == "manual" else None,
            "mapping": "nearest actual PTS; ties choose earlier frame; chronological output",
        }


def parse_timestamps(text: str) -> SelectionSpec:
    try:
        return SelectionSpec.manual([float(t) for t in text.replace(",", " ").split()])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid manual timestamps: {exc}") from exc
