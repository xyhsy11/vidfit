"""Self-contained, escaped results for the paired frame-selection experiment."""

import html
import re


def _escape(value):
    return html.escape(str(value), quote=True)


def _seconds(value):
    return "Unavailable" if value is None else f"{value:,.3f} s"


def _metric(method, key, statistic="mean"):
    stats = method.get("metrics", {}).get(key)
    if not stats or stats.get(statistic) is None:
        return "Unavailable"
    value = stats[statistic]
    return _seconds(value) if key.endswith("_s") else f"{value:,.1f}"


def _reviewed(method, inspection):
    return not inspection and method.get("strict_success_rate") is not None


def _fraction(method, numerator, denominator, inspection):
    if not _reviewed(method, inspection) or method.get(numerator) is None:
        return "Unavailable"
    total = method.get(denominator, 0)
    return f"{method[numerator]} / {total}" if total else "Not applicable"


def _rate(method, inspection):
    if not _reviewed(method, inspection):
        return "Unavailable"
    return f"{100 * method['strict_success_rate']:.1f}%"


def _at(values, index, default=None):
    return values[index] if index < len(values) else default


def _case_link(case, report_by_id):
    # Only the runner's relative report layout is a valid clickable destination.
    path = case.get("report") or report_by_id.get(case["question_id"], "")
    if not isinstance(path, str) or not re.fullmatch(r"case-\d+/comparison\.(json|html)", path):
        return ""
    href = path.rsplit(".", 1)[0] + ".html"
    return f'<a href="{_escape(href)}">Open answers and original source frames</a>'


def _case_card(method, case, inspection):
    times = ", ".join(f"{timestamp:.3f}" for timestamp in case.get("selected_timestamps_s", []))
    coverage = case.get("event_coverage") or {}
    overlap = coverage.get("overlaps_event")
    coverage_text = (
        "Not annotated" if overlap is None else ("Event overlap" if overlap else "No event overlap")
    )
    answers = case.get("answers", [])
    statuses = case.get("statuses", [])
    errors = case.get("errors", [])
    verdicts = case.get("verdicts", [])
    timings = case.get("end_to_end_s", [])
    repeats = []
    for index in range(max(len(answers), len(statuses), len(verdicts))):
        answer = _at(answers, index)
        status = _at(statuses, index, "ok" if answer is not None else "unavailable")
        error = _at(errors, index)
        evaluation = _at(verdicts, index, {}) or {}
        verdict = evaluation.get("verdict", "not_evaluated")
        if inspection or verdict == "not_evaluated":
            verdict_text = "Not evaluated"
        else:
            verdict_text = {
                "correct": "Correct",
                "partially_correct": "Partially correct",
                "incorrect": "Incorrect",
                "unanswerable": "Appropriate abstention",
            }.get(verdict, verdict)
            verdict_text += f" ({evaluation.get('method', 'unreviewed')})"
        answer_text = (
            answer
            if answer is not None
            else (
                "Inspection only; no model answer." if inspection else "No model answer available."
            )
        )
        rationale = evaluation.get("rationale")
        reviewer = evaluation.get("reviewer")
        repeats.append(
            '<div class="trial">'
            f"<h4>Repeat {index + 1} · {_escape(status)}</h4>"
            f"<p>Total time: {_seconds(_at(timings, index))}</p>"
            f"<pre class='answer'>{_escape(answer_text)}</pre>"
            f"<p><b>{_escape(verdict_text)}</b>"
            f"{' · ' + _escape(reviewer) if reviewer and not inspection else ''}</p>"
            f"{'<p>' + _escape(rationale) + '</p>' if rationale and not inspection else ''}"
            f"{'<p class=error>' + _escape(error) + '</p>' if error else ''}"
            "</div>"
        )
    return (
        '<section class="method-card">'
        f"<h3>{_escape(method['label'])}</h3>"
        f"<p><b>Selected times:</b> {_escape(times or 'Unavailable')} seconds</p>"
        f"<p><b>Coverage:</b> {coverage_text}. Coverage is not answer correctness.</p>"
        f"<p><b>Median total:</b> {_seconds(case.get('median_end_to_end_s'))}</p>"
        f"{''.join(repeats)}</section>"
    )


def render_experiment(summary):
    """Render observations and explicit judgments; never infer accuracy from completion."""
    experiment = summary["experiment"]
    methods = summary["methods"]
    inspection = experiment.get("backend") == "inspect"
    questions = summary["unique_questions"]
    repeats = experiment["repeats"]
    settings = experiment.get("settings", {})
    model = experiment.get("model", {})
    header = "".join(f"<th>{_escape(method['label'])}</th>" for method in methods)
    rows = []

    def row(label, values):
        rows.append(
            f"<tr><th>{_escape(label)}</th>"
            + "".join(f"<td>{_escape(value)}</td>" for value in values)
            + "</tr>"
        )

    row("Strict success", [_rate(method, inspection) for method in methods])
    row(
        "Correct supported answers",
        [
            _fraction(method, "supported_correct", "supported_trials", inspection)
            for method in methods
        ],
    )
    row(
        "Appropriate abstentions / unsupported questions",
        [
            _fraction(method, "unsupported_abstentions", "unsupported_trials", inspection)
            for method in methods
        ],
    )
    row(
        "Partially correct answers (no full credit)",
        [
            str(method.get("verdict_counts", {}).get("partially_correct", 0))
            if _reviewed(method, inspection)
            else "Unavailable"
            for method in methods
        ],
    )
    row("Failed requests", [method.get("failures", 0) for method in methods])
    row("Mean total request time", [_metric(method, "end_to_end_s") for method in methods])
    row(
        "Median total request time",
        [_metric(method, "end_to_end_s", "median") for method in methods],
    )
    row("Mean input preparation", [_metric(method, "input_preparation_s") for method in methods])
    row(
        "Mean visual-change scan (included above)",
        [_metric(method, "visual_change_scan_s") for method in methods],
    )
    row("Mean actual visual tokens", [_metric(method, "visual_tokens") for method in methods])
    row("Mean generated tokens", [_metric(method, "generated_tokens") for method in methods])
    row(
        "Successful timed requests",
        [(method.get("metrics", {}).get("end_to_end_s") or {}).get("n", 0) for method in methods],
    )
    row(
        "Answer judgment methods",
        [
            "Not evaluated"
            if inspection
            else ", ".join(method.get("judgment_methods", [])) or "Not evaluated"
            for method in methods
        ],
    )

    report_by_id = {case["question_id"]: case["report"] for case in experiment.get("cases", [])}
    question_order = list(
        dict.fromkeys(case["question_id"] for method in methods for case in method.get("cases", []))
    )
    details = []
    for question_id in question_order:
        cases = [
            (
                method,
                next((c for c in method.get("cases", []) if c["question_id"] == question_id), None),
            )
            for method in methods
        ]
        case = next(case for _, case in cases if case is not None)
        question = case.get("question", "")
        title = f"{question_id} · {question}" if question else question_id
        details.append(
            '<details class="case">'
            f"<summary>{_escape(title)}</summary>"
            f"<p>{_escape(case.get('answerability', 'unspecified'))} question · "
            f"{_case_link(case, report_by_id)}</p>"
            '<div class="case-methods">'
            + "".join(_case_card(method, c, inspection) for method, c in cases if c is not None)
            + "</div></details>"
        )
    notice = experiment.get(
        "evaluation_notice", "Exploratory experiment; interpret within its test set."
    )
    accuracy_definition = summary.get(
        "accuracy_definition", "Accuracy requires explicit answer judgments."
    )
    warmups = experiment.get("warmups", [])
    warmup_values = [(trial.get("metrics") or {}).get("end_to_end_s") for trial in warmups]
    warmup_time = (
        sum(warmup_values) if warmup_values and all(v is not None for v in warmup_values) else None
    )
    timing_contract = experiment.get("timing_contract", {})
    timing_notes = "".join(
        f"<li><b>{_escape(key)}:</b> {_escape(value)}</li>"
        for key, value in timing_contract.items()
    )
    inspection_notice = (
        "<p class=notice><b>Inspection only:</b> no model inference or answer accuracy "
        "is available.</p>"
        if inspection
        else ""
    )
    uniform_definition = experiment.get("uniform_definition", "VidFit time-bin midpoints")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>VidFit · Frame selection experiment</title><style>{_CSS}</style></head><body>"
        '<main><p class="eyebrow">VIDFIT · REAL INPUTS, EXPLICIT JUDGMENTS</p>'
        "<h1>Visual changes vs uniform frames</h1>"
        "<p class='lead'>Four frames selected by visual changes, compared with four and sixteen "
        "uniformly spaced frames.</p>"
        f"<p><b>{questions} unique questions</b> × {repeats} measured repeats per method. "
        "Repeats are observations of the same questions, not additional independent cases.</p>"
        f"<p class='notice'>{_escape(notice)}</p>"
        f"{inspection_notice}"
        '<div class="run-facts">'
        f"<span><b>Backend</b><br>{_escape(experiment.get('backend', 'Unavailable'))}</span>"
        f"<span><b>Model startup</b><br>{_seconds(experiment.get('startup_s'))}</span>"
        f"<span><b>Recorded warmups</b><br>{len(warmups)} · {_seconds(warmup_time)}</span>"
        "<span><b>Recorded experiment wall time</b><br>"
        f"{_seconds(experiment.get('run_elapsed_s'))}</span>"
        "</div>"
        "<h2>Accuracy and runtime</h2>"
        f"<p>{_escape(accuracy_definition)}</p>"
        "<p>Aggregate accuracy remains unavailable until completed answers have explicit "
        "judgments. "
        "Agent judgments are not independent human review. Failed requests count against strict "
        "success when accuracy is available; runtime averages use successful requests only.</p>"
        '<div class="table-scroll" tabindex="0" aria-label="Method comparison">'
        f"<table><thead><tr><th>Measure</th>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        "<h2>What the timer includes</h2>"
        "<p>Total request time starts before probing and hashing the video, and includes "
        "selection, "
        "source decoding and evidence capture, model preprocessing, device transfer, inference "
        "and output decoding. The loaded model is reused. Startup and warmups are recorded "
        "separately; exports and answer review are outside the per-request timer.</p>"
        "<p>Visual-change scanning is already included in input preparation; it must not be added "
        "a second time. Output lengths can differ and affect total time.</p>"
        f"<p><b>Uniform baseline:</b> {_escape(uniform_definition)}.</p>"
        "<details><summary>Exact timing contract</summary>"
        f"<ul>{timing_notes}</ul></details>"
        "<h2>Answers and selected evidence</h2>"
        f"{''.join(details)}"
        "<details class='provenance'><summary>Run provenance</summary>"
        f"<p>Model: {_escape(model.get('identifier') or settings.get('model', 'Unavailable'))}<br>"
        f"Resolved revision: {_escape(model.get('resolved_revision', 'Unavailable'))}<br>"
        f"Device / precision: {_escape(settings.get('device', 'Unavailable'))} / "
        f"{_escape(settings.get('dtype', 'Unavailable'))}<br>"
        f"Pixel cap per frame: {_escape(settings.get('max_pixels', 'Unavailable'))}<br>"
        f"Output-token limit: {_escape(settings.get('max_new_tokens', 'Unavailable'))}<br>"
        f"Status: {_escape(experiment.get('status', 'Unavailable'))}<br>"
        f"Run: {_escape(experiment.get('run_id', 'Unavailable'))}<br>"
        f"Created: {_escape(experiment.get('created_at_utc', 'Unavailable'))}</p>"
        "<p><a href='summary.json'>Structured summary JSON</a> · "
        "<a href='experiment.json'>Schedule, warmups and environment</a></p>"
        "</details></main></body></html>"
    )


_CSS = """
*{box-sizing:border-box}body{margin:0;background:#f1f5f3;color:#1e3833;
 font:15px/1.55 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:32px}
h1{font-size:34px;line-height:1.2;margin:8px 0 12px}h2{font-size:23px;margin-top:28px}
h3{font-size:17px;margin:0 0 10px}h4{font-size:14px;margin:10px 0 6px}
p{margin:8px 0 12px}.eyebrow{font-size:12px;letter-spacing:.08em;color:#426c60}
.lead{font-size:18px}.notice{background:#fff5dc;border-left:4px solid #be8b25;padding:14px}
.run-facts{display:flex;flex-wrap:wrap;gap:16px;background:#dfece6;padding:16px;border-radius:8px}
.run-facts span{flex:1;min-width:175px}
.table-scroll{overflow:auto;background:white;border:1px solid #bfd1c8}
table{border-collapse:collapse;width:100%;min-width:760px}
th,td{padding:11px 14px;border-bottom:1px solid #dbe5df;
 text-align:left}thead th{background:#244f42;color:white}tbody th{font-weight:550;width:34%}
tbody tr:nth-child(even){background:#f5f9f6}td{font-variant-numeric:tabular-nums}
details{margin:14px 0}summary{font-weight:650;cursor:pointer}.case{padding:15px;background:white;
 border:1px solid #bfd1c8;border-radius:8px}.case summary{font-size:17px}
.case-methods{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.method-card{min-width:0;background:#f6f9f7;border:1px solid #d3dfd8;padding:14px;border-radius:6px}
.method-card p{font-size:13px;overflow-wrap:anywhere}
.trial{border-top:1px solid #cbdad1;padding-top:5px;margin-top:15px}
.answer{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.5 system-ui,sans-serif;
 background:white;
 padding:10px;border-left:3px solid #649380}.error{color:#9c2b28}.provenance{overflow-wrap:anywhere}
a{color:#12684d;text-underline-offset:3px}@media(max-width:1050px){.case-methods{grid-template-columns:1fr}}
@media(max-width:600px){main{padding:16px}h1{font-size:27px}.run-facts span{min-width:130px}}
"""
