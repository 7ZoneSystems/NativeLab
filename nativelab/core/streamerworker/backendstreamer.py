from nativelab.imports.import_global import QThread, pyqtSignal, time


class BackendStreamWorker(QThread):
    """Stream tokens from any engine backend that exposes generate_sync(...)."""

    token = pyqtSignal(str)
    done = pyqtSignal(float)
    err = pyqtSignal(str)

    def __init__(self, engine, prompt: str, n_predict: int, **kwargs):
        super().__init__()
        self.engine = engine
        self.prompt = prompt
        self.n_predict = int(n_predict)
        self.kwargs = dict(kwargs)
        self._abort = False

    def run(self):
        t0 = time.time()
        n = 0

        def _on_token(tok: str):
            nonlocal n
            if self._abort:
                return
            if tok:
                n += 1
                self.token.emit(str(tok))

        try:
            self.engine.generate_sync(
                prompt=self.prompt,
                n_predict=self.n_predict,
                token_cb=_on_token,
                abort_cb=lambda: self._abort,
                **self.kwargs,
            )
        except Exception as exc:
            self.err.emit(f"{type(exc).__name__}: {exc}")
            return
        elapsed = time.time() - t0
        self.done.emit(n / elapsed if elapsed > 0 else 0.0)

    def abort(self):
        self._abort = True
