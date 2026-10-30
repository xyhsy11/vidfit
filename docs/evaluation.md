# Starter evaluation workflow

This collection exercises selection and evaluation plumbing. Five short clips and
ten questions are far too small and deliberately chosen to claim general model
accuracy. References were **visually reviewed by Codex, an agent**, on 2026-10-30.
They have not been independently human-reviewed. Model answers are not generated
from the references, and reference review does not evaluate a model answer.

## Prepare and verify media

From the repository root, after installing VidFit:

```powershell
.\.venv\Scripts\python.exe -m pip install av==16.0.1 Pillow==12.0.0 numpy==2.2.6
.\.venv\Scripts\python.exe examples/prepare_evaluation.py
.\.venv\Scripts\vidfit.exe evaluate --collection examples/evaluation/starter.json --video-dir examples/generated/evaluation --backend inspect --frames 4 --output-dir reports/starter
```

On Unix use `python examples/prepare_evaluation.py` and `vidfit evaluate ...` with
the same arguments. The script downloads originals into `.cache/evaluation`,
checks their SHA-256 values, and builds clips in `examples/generated/evaluation`.
It prints output checksums and checks them against the versioned annotations.
Existing media is reused and verified, not blindly overwritten. Reports, original
media and generated media remain ignored by Git.

Encoding is reproducible in the reference environment: PyAV 16.0.1, Pillow 12.0.0,
NumPy 2.2.6 and the FFmpeg library versions in the validation record. Different
encoder/font/library versions can change bytes. On a mismatch, use the reference
environment or visually review and version your own references with their new
hashes; do not disable identity validation.

| Clip | Length | Content and questions | Provenance / permission |
| --- | --- | --- | --- |
| 01 | 60 s | Application synchronization; brief error and task-name questions | VidFit synthetic generator, MIT |
| 02 | 6 s | Blue square moves right; direction and unsupported operator-name questions | VidFit synthetic generator, MIT |
| 03 | 6 s | Green background with orange flash at `[2,2.25)`; color and unsupported sign-text questions | VidFit synthetic generator, MIT |
| 04 | 8 s | Cat on sofa; animal and unsupported name questions | Ivan Kurmanov, CC BY 2.0; first 8 s, silent, resized / re-encoded |
| 05 | 4.5 s | Earth rotates with visible clouds; motion and cloud questions | NASA/GSFC public-domain imagery; source includes time-lapse interpolation; silent, resized / re-encoded |

The real footage sources are [Cat relaxing on a sofa](https://commons.wikimedia.org/wiki/File:Cat_relaxing_on_a_sofa.webm)
by Ivan Kurmanov ([CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)) and
[DSCOVR EPIC Earth Rotation](https://commons.wikimedia.org/wiki/File:DSCOVR_EPIC_Earth_Rotation.webm)
by NASA/GSFC, uploaded by Edits4019. The latter description identifies NASA public
domain status in the United States and DAIN interpolation; it is derived satellite
time-lapse imagery, not a real-time recording. Source/download URLs, attribution,
changes and original/output SHA-256 values are in
[starter.json](../examples/evaluation/starter.json) and the preparation script.
No upstream video bytes are distributed in this repository.

Visual review used actual decoded RGB frames at the timestamps recorded in each
annotation's `review.inspected_timestamps_s`. For synthetic events, boundary frames
immediately before, inside and after the interval were inspected against generator
source. Cat review inspected 0, 1, 2, 4, 6 and 7.875 s of the eight-second derivative;
Earth review inspected 0, 0.5, 1.5, 2.5, 3.5 and 4.375 s. This is a documented agent
spot-check, not exhaustive frame-by-frame human certification.

## Annotation contract

A standalone question uses `schema: vidfit.annotation.v1`; a collection uses
`schema: vidfit.collection.v1` with an `annotations` array. Each annotation contains:

- `id`, `video.filename`, `video.sha256` and synthetic/real identity;
- `source.uri`, `source.license`, attribution, and optional download/license links;
- `question`, `accepted_answers` and a scoring `rubric`;
- relevant `intervals_s` as half-open pairs (empty for unsupported/no localized event);
- `answerability` (`supported` or `unsupported`) and a `review` status with provenance.

Review status is `unreviewed`, `agent_visual_review` or `human_reviewed`. Only a
person's actual review should be recorded as human-reviewed. A reference must match
the input video's hash and question exactly. Invalid intervals and mismatched files
are rejected before backend loading.

To run one referenced question with Qwen:

```powershell
.\.venv\Scripts\vidfit.exe compare --video examples/generated/evaluation/clip-01.mp4 --question "What error message appeared?" --timestamps 7.5 24.375 37.5 52.5 --backend qwen3vl --execution-mode single --annotation examples/evaluation/starter.json --annotation-id error-text --output-dir reports/eval-question
```

In the UI, upload the JSON in **Evaluation and benchmark options** and supply the
question ID for a collection. The synthetic `.truth.json` format is still accepted
there for evidence overlap only. References and rubrics never enter the model input;
CPU tests assert this at the backend boundary. `evaluate --backend qwen3vl` runs all
ten questions with uniform selection, one invocation each, leaving answers unreviewed.
It reloads the backend per question; this is a functional runner, not an aggregate
throughput benchmark.

## Record judgments

A completed inference is initially **Not evaluated (unreviewed)**. Read the answer
and inputs, consult the independent rubric, then record an explicit judgment:

```powershell
.\.venv\Scripts\vidfit.exe judge --report reports/eval-question/comparison.json --configuration 0 --trial 0 --verdict partially_correct --method agent --reviewer Codex --rationale "Includes E104 and timeout but mixes in unrelated status text." --output-dir reports/eval-question-reviewed
```

Use `--method human` only for an actual human judgment and `--method automated`
for a named programmatic check. Agent judgments remain `agent`; this release does
not silently score answers or claim a human review. Verdicts are `correct`,
`partially_correct`, `incorrect`, and `unanswerable`. The last means an appropriate
recognition that the question is unsupported, not a failed run or missing output.
Each judgment needs a short rationale and reviewer/check identifier and is bound
to the answer hash. Indices are zero-based trial-list indices; in benchmark mode,
first-invocation/warmup entries precede measured entries. Inspect JSON before judging.
The command exports a new HTML/JSON pair and refuses to judge failed/inspection-only
trials with no answer.

Evaluate evidence coverage, answer quality and costs separately. Overlap is any
selected timestamp inside an annotation interval. It does not guarantee that pixels
are sufficient, that the model used them, or that the answer is correct. Unsupported
questions deliberately test whether a model invents information absent from video.
Broader evaluation and independent human review are follow-up milestones.
