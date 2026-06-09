"""Qt worker lifecycle helpers.

Centralizing this keeps shutdown behavior consistent across MainWindow tabs and
reduces the chance of destroying widgets while their QThread callbacks are
still alive.
"""
from __future__ import annotations

from typing import Iterable


COMMON_SIGNAL_NAMES = (
    "finished",
    "done",
    "err",
    "error",
    "token",
    "progress",
    "status",
    "paused",
    "results_ready",
    "code_ready",
    "plan_ready",
    "step_started",
    "step_token",
    "step_done",
    "intermediate_live",
    "pipeline_done",
    "log_msg",
    "section_done",
    "final_done",
    "file_started",
    "file_progress",
    "file_done",
    "ram_warning",
    "pause_suggest",
)


def qthread_running(worker) -> bool:
    if worker is None:
        return False
    try:
        return bool(worker.isRunning())
    except Exception:
        return False


def disconnect_worker_signals(worker, names: Iterable[str] = COMMON_SIGNAL_NAMES) -> None:
    if worker is None:
        return
    for name in names:
        sig = getattr(worker, name, None)
        if sig is None or not hasattr(sig, "disconnect"):
            continue
        try:
            sig.disconnect()
        except Exception:
            pass


def stop_worker(
    worker,
    timeout_ms: int,
    *,
    abort: bool = True,
    cancel: bool = True,
    request_pause: bool = False,
    delete_part: bool | None = None,
    disconnect_on_stopped: bool = True,
) -> bool:
    """Ask a QThread-like worker to stop and wait for it.

    Returns True when the worker is stopped or was never running. It intentionally
    does not call QThread.terminate(); terminating native/model-loading work is
    more likely to cause the segmentation faults this helper is meant to avoid.
    """
    if worker is None:
        return True

    if abort and hasattr(worker, "abort"):
        try:
            if delete_part is None:
                worker.abort()
            else:
                worker.abort(delete_part=delete_part)
        except TypeError:
            try:
                worker.abort()
            except Exception:
                pass
        except Exception:
            pass
    elif cancel and hasattr(worker, "cancel"):
        try:
            worker.cancel()
        except Exception:
            pass
    elif request_pause and hasattr(worker, "request_pause"):
        try:
            worker.request_pause()
        except Exception:
            pass

    if hasattr(worker, "quit"):
        try:
            worker.quit()
        except Exception:
            pass

    stopped = True
    if qthread_running(worker) and hasattr(worker, "wait"):
        try:
            stopped = bool(worker.wait(int(timeout_ms)))
        except Exception:
            stopped = not qthread_running(worker)
    else:
        stopped = not qthread_running(worker)

    if stopped and disconnect_on_stopped:
        disconnect_worker_signals(worker)
    return bool(stopped)


def stop_worker_attrs(owner, attr_names: Iterable[str], timeout_ms: int, **kwargs) -> list[str]:
    still_running: list[str] = []
    for attr in attr_names:
        worker = getattr(owner, attr, None)
        if worker is None:
            continue
        if stop_worker(worker, timeout_ms, **kwargs):
            try:
                setattr(owner, attr, None)
            except Exception:
                pass
        else:
            still_running.append(attr)
    return still_running
