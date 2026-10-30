from vidfit.core.profiling import INFERENCE_METRICS

INSPECTION_NOTICE = "Inspection only — model inference was not run."


class InspectionBackend:
    info = {
        "identifier": None,
        "processor_identifier": None,
        "resolved_revision": None,
        "loading_s": None,
        "loading_unavailable_reason": INSPECTION_NOTICE,
    }

    def synchronize(self):
        pass

    def reset_peak_memory(self):
        pass

    def run(self, frames, metadata, question):
        return {
            "answer": None,
            "metrics": dict.fromkeys(INFERENCE_METRICS),
            "unavailable_reasons": dict.fromkeys((*INFERENCE_METRICS, "answer"), INSPECTION_NOTICE),
            "processing": {
                "status": INSPECTION_NOTICE,
                "selected_frames": len(frames),
                "model_input_frames": None,
            },
        }
