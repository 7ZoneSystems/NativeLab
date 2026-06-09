"""Background loader threads used by the GUI."""
from nativelab.imports.import_global import QThread, pyqtSignal
from nativelab.core.engine_global import ApiEngine


class ModelLoaderThread(QThread):
    finished = pyqtSignal(bool, str)
    log = pyqtSignal(str, str)

    def __init__(self, engine, model_path: str, ctx: int, threads: int):
        super().__init__()
        self.engine = engine
        self.model_path = model_path
        self.ctx = ctx
        self.threads = threads

    def run(self):
        ok = self.engine.load(
            self.model_path,
            threads=self.threads,
            ctx=self.ctx,
            log_cb=lambda m: self.log.emit("INFO", str(m)),
        )
        self.finished.emit(ok, self.engine.status_text)


class ApiLoaderThread(QThread):
    finished = pyqtSignal(bool, str, object)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        eng = ApiEngine()
        ok = eng.load(self.config)
        self.finished.emit(
            ok,
            eng.status_text if ok else "API connection failed",
            eng if ok else None,
        )
