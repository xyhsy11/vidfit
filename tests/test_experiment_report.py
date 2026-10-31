"""Reports preserve explicit review states, timing boundaries and safe offline links."""

from copy import deepcopy

import pytest

from vidfit.evaluation.experiment_report import render_experiment


@pytest.fixture
def reviewed_summary():
    def case(question_id, supported):
        return {
            "question_id": question_id,
            "question": "What changed?" if supported else "Who operated the device?",
            "answerability": "supported" if supported else "unsupported",
            "selected_timestamps_s": [0.125, 1.25, 2.5, 3.75],
            "event_coverage": {"overlaps_event": True if supported else None},
            "answers": ["The display turned red." if supported else "The operator is not shown."],
            "verdicts": [
                {
                    "verdict": "correct" if supported else "unanswerable",
                    "method": "agent",
                    "reviewer": "Test reviewer",
                    "rationale": "Matches the rubric.",
                }
            ],
            "statuses": ["ok"],
            "errors": [None],
            "end_to_end_s": [1.25],
            "median_end_to_end_s": 1.25,
        }

    method = {
        "method": "visual_change_4",
        "label": "Visual changes · 4 frames",
        "judgment_methods": ["agent"],
        "verdict_counts": {
            "correct": 1,
            "partially_correct": 0,
            "unanswerable": 1,
            "incorrect": 0,
            "not_evaluated": 0,
        },
        "failures": 0,
        "supported_trials": 1,
        "unsupported_trials": 1,
        "supported_correct": 1,
        "unsupported_abstentions": 1,
        "strict_success_rate": 1,
        "metrics": {
            key: {"mean": value, "median": value, "n": 2, "min": value, "max": value}
            for key, value in [
                ("end_to_end_s", 1.25),
                ("input_preparation_s", 0.25),
                ("visual_change_scan_s", 0.125),
                ("visual_tokens", 440),
                ("generated_tokens", 12),
            ]
        },
        "cases": [case("brief-event", True), case("operator", False)],
    }
    uniform4, uniform16 = deepcopy(method), deepcopy(method)
    uniform4.update(method="uniform_4", label="Uniform midpoints · 4 frames")
    uniform16.update(method="uniform_16", label="Uniform midpoints · 16 frames")
    uniform4["metrics"]["visual_change_scan_s"] = None
    uniform16["metrics"]["visual_change_scan_s"] = None
    return {
        "experiment": {
            "backend": "qwen3vl",
            "repeats": 1,
            "startup_s": 3,
            "run_elapsed_s": 13,
            "status": "complete",
            "run_id": "test-run",
            "created_at_utc": "2026-10-31",
            "settings": {
                "model": "Qwen/test",
                "device": "cuda:0",
                "dtype": "float16",
                "max_pixels": 262144,
                "max_new_tokens": 64,
            },
            "model": {"identifier": "Qwen/test", "resolved_revision": "test-revision"},
            "warmups": [{"metrics": {"end_to_end_s": 1}}, {"metrics": {"end_to_end_s": 1.5}}],
            "uniform_definition": (
                "VidFit uniform time-bin midpoints, not native Qwen endpoint sampling"
            ),
            "evaluation_notice": (
                "Known starter clips; agent-reviewed references, not a held-out benchmark."
            ),
            "timing_contract": {"end_to_end_s": "Includes preprocessing and selection."},
            "cases": [
                {"question_id": "brief-event", "report": "case-000/comparison.json"},
                {"question_id": "operator", "report": "case-001/comparison.json"},
            ],
        },
        "unique_questions": 2,
        "methods": [method, uniform4, uniform16],
        "accuracy_definition": (
            "Correct supported answer or appropriate abstention; no partial credit."
        ),
    }


def test_reviewed_report_has_accuracy_costs_and_case_evidence(reviewed_summary):
    text = render_experiment(reviewed_summary)
    assert "100.0%" in text and "1 / 1" in text
    assert "Correct supported answers" in text and "Appropriate abstentions" in text
    assert "Mean total request time" in text and "Median total request time" in text
    assert "Mean input preparation" in text and "0.250 s" in text
    assert "440.0" in text and "12.0" in text
    assert "2 unique questions" in text and "not additional independent cases" in text
    assert "3.000 s" in text and "13.000 s" in text and "2.500 s" in text
    assert "The display turned red." in text and "Matches the rubric." in text
    assert "0.125, 1.250, 2.500, 3.750" in text
    assert 'href="case-000/comparison.html"' in text
    assert "not native Qwen endpoint sampling" in text
    assert "Agent judgments are not independent human review" in text
    assert "already included in input preparation" in text
    assert "<script" not in text and 'src="http' not in text and 'href="http' not in text


@pytest.mark.parametrize("inspection", [False, True])
def test_accuracy_is_unavailable_until_reviewed_and_always_for_inspection(
    reviewed_summary, inspection
):
    if inspection:
        reviewed_summary["experiment"]["backend"] = "inspect"
    else:
        for method in reviewed_summary["methods"]:
            method["strict_success_rate"] = None
            method["supported_correct"] = None
            method["unsupported_abstentions"] = None
            method["judgment_methods"] = ["unreviewed"]
            for case in method["cases"]:
                case["verdicts"] = [{"verdict": "not_evaluated", "method": "unreviewed"}]
    text = render_experiment(reviewed_summary)
    assert "100.0%" not in text and "<td>1 / 1</td>" not in text
    assert "Unavailable" in text and "Not evaluated" in text
    if inspection:
        assert "Inspection only" in text
        assert "Correct (agent)" not in text


def test_failed_case_retains_error_and_unavailable_metrics(reviewed_summary):
    method = reviewed_summary["methods"][0]
    method.update(failures=1, strict_success_rate=0.5, supported_correct=0)
    method["metrics"] = {}
    case = method["cases"][0]
    case.update(
        answers=[None],
        statuses=["failed"],
        errors=["CUDA out of memory"],
        verdicts=[{"verdict": "not_evaluated", "method": "unreviewed"}],
        end_to_end_s=[0.75],
        median_end_to_end_s=None,
    )
    text = render_experiment(reviewed_summary)
    assert "Repeat 1 · failed" in text and "CUDA out of memory" in text
    assert "No model answer available" in text and "0.750 s" in text
    assert "50.0%" in text and "Failed requests" in text


def test_untrusted_text_is_escaped_and_report_paths_are_restricted(reviewed_summary):
    payload = '<script>alert("unsafe")</script>'
    reviewed_summary["experiment"]["evaluation_notice"] = payload
    reviewed_summary["experiment"]["cases"][0]["report"] = "javascript:alert(1)"
    for method in reviewed_summary["methods"]:
        method["label"] = payload
        case = method["cases"][0]
        case["question"] = payload
        case["answers"] = [payload]
        case["verdicts"][0]["rationale"] = payload
        case["report"] = "../private/comparison.html"
    text = render_experiment(reviewed_summary)
    assert payload not in text and "&lt;script&gt;" in text
    assert 'href="javascript:' not in text and 'href="../' not in text
    assert 'href="case-001/comparison.html"' in text
