# Visual-change selection: real Qwen pilot (2026-10-31)

The fixed visual-change selector did **not improve strict answer accuracy over
uniform 4** on this starter set. It was faster on average than uniform 16, but
uniform 16 answered one more of the ten questions correctly. These are observations
from five known clips, not a held-out benchmark or evidence of a general speedup.

## Results

Ten questions, two measured repeats per method: 60 real Qwen requests, all successful.
All answers were identical across repeats within each question/method. The repeated
requests provide timing observations; they are not 20 independent questions.

| Method | Strict success per unique question | Correct supported answers | Appropriate abstentions | Mean total request | Median total request | Mean input preparation |
| --- | --- | --- | --- | --- | --- | --- |
| Visual changes, 4 frames | 6/10 (60%) | 4/7 | 2/3 | 2.528 s | 1.816 s | 0.645 s |
| Uniform midpoints, 4 frames | 6/10 (60%) | 4/7 | 2/3 | 2.576 s | 1.640 s | 0.450 s |
| Uniform midpoints, 16 frames | 7/10 (70%) | 5/7 | 2/3 | 3.574 s | 2.472 s | 0.839 s |

Strict success requires a fully correct supported answer or an appropriate abstention
on an unsupported question. Partial answers receive no full credit. Supported-answer
accuracy alone is 57.1%, 57.1%, and 71.4%, respectively. The machine-readable reports
count both repeats, so their denominators are 20, 14, and 6 rather than 10, 7, and 3.

Visual-change selection adds a mean 0.223 s scan, already included in the totals.
Its mean total is 29.3% lower than uniform 16 in this run, with lower strict accuracy.
The 0.048 s mean advantage over uniform 4 is small and reverses in the median.
Mean generated-token counts also differ: 12.6, 16.2, and 17.3. These timings include
different answer lengths and do not isolate visual encoding cost.

Both four-frame methods used exactly the same visual-token count for each source;
uniform 16 used four times as many. Across the collection, mean actual visual-token
counts were 314, 314, and 1256. These were measured from Qwen's processed inputs,
not estimated from frame counts.

## What changed in the answers

| Question | Visual changes 4 | Uniform 4 | Uniform 16 |
| --- | --- | --- | --- |
| Error message | Partial | Incorrect | Correct |
| Application task | Correct | Correct | Correct |
| Square direction | Correct | Correct | Correct |
| Operator name, unsupported | Abstains | Abstains | Abstains |
| Brief replacement color | Incorrect | Incorrect | Incorrect |
| Sign text, unsupported | Incorrect | Incorrect | Incorrect |
| Animal on sofa | Correct | Correct | Correct |
| Cat name, unsupported | Abstains | Abstains | Abstains |
| Earth motion | Incorrect | Partial | Partial |
| Clouds visible | Correct | Correct | Correct |

The error-screen example shows why selected evidence and answer correctness must
remain separate. Visual-change 4 selected 8.0, 24.0, 31.25, and 47.0 seconds,
including the error interval. Qwen answered `Error E104: Processing records...`:
the code is correct, but the required message is `Connection timed out.` Uniform 4
missed the interval and denied an error. Uniform 16 returned the correct full message.

Visual-change 4 also captured the brief orange frame, yet all three methods answered
white. All three invented text for a nonexistent sign. Earth-motion answers included
false zoom, cropping, or continent-order claims. The partial-versus-incorrect Earth
judgments were checked against all 36 source frames, independently of method labels.
There was no post-result selector tuning.

## Fixed method and execution

The question-independent selector downsizes every decoded frame to 64 x 36 RGB,
scores mean absolute difference from the preceding frame, and picks the largest
positive change in each of four equal-duration bins. Earlier frames win score ties;
static bins use their nearest midpoint. A second decode captures the exact selected
frames in chronological order. Selection receives no question, reference answer, or
event annotation. Uniform baselines use VidFit's time-bin midpoints; they are not
Qwen's native endpoint sampler.

Run `e2fdcadc-d776-4894-bb4a-f3dfc244c72a` used Windows, Python 3.12.14,
PyTorch 2.8.0+cu128, Transformers 4.57.6, and an RTX 3080 Laptop GPU with 16 GB.
Model: `Qwen/Qwen3-VL-2B-Instruct`, resolved model/processor revision
`89644892e4d85e24eaac8bacfd4f463576704203`; float16, SDPA, greedy decoding,
64-new-token limit, 262144-pixel per-frame cap. Cached weights loaded offline.
One model was reused, two warmups were recorded, and six method-order permutations
were cycled. Selected frame indices, timestamps, hashes, native inputs, and absence
of resampling/padding were verified for every measured request.

Each request timer starts before metadata probing and hashing. It includes full
source decoding, the selector scan and capture, PNG evidence encoding/hashing,
Qwen preprocessing, device transfer, visual encoding, language generation, output
text decoding, and final GPU synchronization. Input preparation is the subset before
generation; nested scan/capture times are never added twice. This implementation
includes workbench evidence-capture overhead and is not an optimized serving pipeline.

One-time runtime/model startup was 15.041 s, including 8.744 s model loading.
The two warmups totaled 6.717 s. Measured requests totaled 173.570 s. Experiment wall
time was 197.504 s, including startup, warmups and per-case exports, but excluding
later answer review. These separate costs are not silently omitted or multiplied
into the warm-model per-request values.

## Review and reproduction

Existing starter references have agent visual review. A separate agent reviewed
21 deduplicated question/answer pairs in shuffled order with method, frame selection,
and timing hidden. Explicit verdicts and rationales were applied by answer hash.
This is agent judgment against a small known set, not independent human evaluation.

Follow the [experiment instructions](../README.md#compare-a-visual-change-selector-with-uniform-sampling).
For exact model identity, add
`--revision 89644892e4d85e24eaac8bacfd4f463576704203` to that command.
Use a fresh output directory. Accuracy remains unreviewed until explicit judgments
are applied; successful execution does not assign a correctness score.

Local outputs are under `reports/visual-change-20261031/`: `summary.html` and JSON,
ten case reports with embedded selected frames, untouched `comparison.raw.json`
records, the blinded review packet, and `judgments.json`. The four measured source
files are preserved in `measured-source/` and match the manifest fingerprints.
A post-run fix only makes optional reference fields use their schema defaults;
the measured selector, model backend, and timing logic are unchanged.

The next useful research experiment would use more unseen clips and independently
reviewed answers, and compare temporal context around detected changes. This pilot
establishes a reproducible baseline, not a demonstrated accuracy improvement.
