"""Optional evaluation references and judgments. Never imported by a model backend."""

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

VERDICTS = ("correct", "partially_correct", "incorrect", "unanswerable")
METHODS = ("human", "automated", "agent")


def validate_annotation(item):
    if not isinstance(item, dict):
        raise ValueError("annotation must be a JSON object")
    if item.get("schema") != "vidfit.annotation.v1":
        raise ValueError("annotation schema must be vidfit.annotation.v1")
    for field in ("id", "question", "rubric"):
        if not isinstance(item.get(field), str) or not item[field].strip():
            raise ValueError(f"annotation needs nonempty {field}")
    video = item.get("video", {})
    if not isinstance(video, dict):
        raise ValueError("annotation video must be an object")
    if (
        not isinstance(video.get("filename"), str)
        or not isinstance(video.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", video.get("sha256", ""))
    ):
        raise ValueError("annotation needs video filename and SHA-256")
    source = item.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("annotation source must be an object")
    for field in ("uri", "license", "attribution"):
        if not isinstance(source.get(field), str) or not source[field].strip():
            raise ValueError(f"annotation source needs {field}")
    review = item.get("review", {})
    if not isinstance(review, dict):
        raise ValueError("annotation review must be an object")
    if item.get("answerability", "supported") not in ("supported", "unsupported"):
        raise ValueError("answerability must be supported or unsupported")
    if review.get("status") not in ("unreviewed", "agent_visual_review", "human_reviewed"):
        raise ValueError("annotation review status must identify unreviewed, agent or human review")
    if review["status"] != "unreviewed" and any(
        not isinstance(review.get(k), str) or not review[k].strip()
        for k in ("reviewer", "date", "provenance")
    ):
        raise ValueError("reviewed annotations need reviewer, date and provenance")
    if not isinstance(item.get("accepted_answers", []), list) or any(
        not isinstance(a, str) for a in item.get("accepted_answers", [])
    ):
        raise ValueError("accepted_answers must be a list of strings")
    if not isinstance(item.get("intervals_s", []), list):
        raise ValueError("intervals_s must be a list of half-open intervals")
    for interval in item.get("intervals_s", []):
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(
                isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
                for t in interval
            )
            or not 0 <= interval[0] < interval[1]
        ):
            raise ValueError("annotation intervals must be finite, increasing and nonnegative")
    return item


def read_collection(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("schema") != "vidfit.collection.v1"
        or not isinstance(data.get("annotations"), list)
        or not data["annotations"]
    ):
        raise ValueError("expected a nonempty vidfit.collection.v1 annotation collection")
    items = [validate_annotation(item) for item in data["annotations"]]
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("annotation IDs must be unique")
    return items


def load_annotation(path, metadata, question, annotation_id=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("annotation must be a JSON object")
    items = read_collection(path) if data.get("schema") == "vidfit.collection.v1" else [data]
    matches = [i for i in items if i.get("id") == annotation_id] if annotation_id else items
    if len(matches) != 1:
        raise ValueError("choose exactly one annotation with annotation_id")
    item = validate_annotation(matches[0])
    if item["video"]["sha256"] != metadata["sha256"]:
        raise ValueError("annotation must match the video SHA-256")
    if item["question"] != question:
        raise ValueError("annotation question must exactly match the model question")
    if any(b > metadata["duration_s"] + 1e-6 for _, b in item.get("intervals_s", [])):
        raise ValueError("annotation interval exceeds video duration")
    return item


def evidence_coverage(annotation, frames):
    intervals = annotation.get("intervals_s", [])
    hits = [f.frame_index for f in frames if any(a <= f.timestamp_s < b for a, b in intervals)]
    return {
        "overlaps_event": bool(hits) if intervals else None,
        "overlapping_frame_indices": hits,
        "definition": "actual selected PTS in annotated half-open intervals; not answer accuracy",
        "annotation_review_status": annotation["review"]["status"],
    }


def record_judgment(result, configuration, trial_index, verdict, method, rationale, reviewer):
    if verdict not in VERDICTS or method not in METHODS:
        raise ValueError("invalid verdict or judgment method")
    if not rationale.strip() or not reviewer.strip():
        raise ValueError("judgments require a short rationale and reviewer/check identifier")
    if not 0 <= configuration < len(result.configurations):
        raise ValueError("configuration index out of range")
    trials = result.configurations[configuration].trials
    if not 0 <= trial_index < len(trials):
        raise ValueError("trial index out of range")
    trial = trials[trial_index]
    if trial.status != "ok" or trial.answer is None:
        raise ValueError("cannot judge a failed or inspection-only trial with no model answer")
    trial.evaluation = {
        "verdict": verdict,
        "method": method,
        "rationale": rationale,
        "reviewer": reviewer,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "answer_sha256": hashlib.sha256(trial.answer.encode("utf-8")).hexdigest(),
    }
    return result
