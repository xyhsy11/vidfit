"""Equal-bin midpoints, in seconds relative to the first presentation timestamp."""

import math


def midpoint_targets(duration_s: float, frame_budget: int) -> list[float]:
    if not isinstance(frame_budget, int) or isinstance(frame_budget, bool) or frame_budget < 1:
        raise ValueError("frame budget must be a positive integer")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("video duration must be finite and positive")
    return [(i + 0.5) * duration_s / frame_budget for i in range(frame_budget)]
