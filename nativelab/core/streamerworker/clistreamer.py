from nativelab.imports.import_global import QThread, pyqtSignal, time, subprocess, Optional
class CliStreamWorker(QThread):
    token  = pyqtSignal(str)
    done   = pyqtSignal(float)
    err    = pyqtSignal(str)

    def __init__(self, cmd: list):
        super().__init__()
        self.cmd    = cmd
        self._abort = False
        self.proc:  Optional[subprocess.Popen] = None

    def run(self):
        t0 = time.time(); n = 0
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                bufsize=0
            )
            while not self._abort:
                if self.proc.stdout is None:
                    break
                b = self.proc.stdout.read(1)
                if not b: break
                c = b.decode("utf-8", errors="replace")
                n += 1; self.token.emit(c)
            self.proc.wait(timeout=5)
            elapsed = time.time() - t0
            self.done.emit(n / elapsed if elapsed > 0 else 0.0)
        except Exception as e:
            self.err.emit(str(e))

    def abort(self):
        self._abort = True
        if self.proc:
            try: self.proc.terminate()
            except OSError: pass
