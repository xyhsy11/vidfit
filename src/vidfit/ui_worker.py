"""One local process isolates core work from the UI event loop and its GIL."""

import atexit
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def _ready():
    return True


class LocalWorker:
    def __init__(self, controller):
        self.controller = controller
        self.executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )
        atexit.register(self.close)

    def start(self):
        return self.executor.submit(_ready).result()

    def preview(self, *args, **kwargs):
        return self.executor.submit(self.controller.preview, *args, **kwargs).result()

    def trace(self, *args, **kwargs):
        return self.executor.submit(self.controller.trace, *args, **kwargs).result()

    def run(self, *args, progress=None, **kwargs):
        if progress:
            progress("Running local core job; setup and measured inference are recorded separately")
        return self.executor.submit(self.controller.run, *args, **kwargs).result()

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
