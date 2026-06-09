import unittest

from nativelab.UI.qt_workers import stop_worker, stop_worker_attrs


class _Signal:
    def __init__(self):
        self.disconnected = False

    def disconnect(self):
        self.disconnected = True


class _Worker:
    def __init__(self, stops=True):
        self.running = True
        self.stops = stops
        self.aborted = False
        self.quit_called = False
        self.finished = _Signal()

    def isRunning(self):
        return self.running

    def abort(self):
        self.aborted = True

    def quit(self):
        self.quit_called = True

    def wait(self, _timeout):
        if self.stops:
            self.running = False
        return not self.running


class QtWorkerLifecycleTests(unittest.TestCase):
    def test_stop_worker_aborts_waits_and_disconnects_when_stopped(self):
        worker = _Worker()
        self.assertTrue(stop_worker(worker, 10))
        self.assertTrue(worker.aborted)
        self.assertTrue(worker.quit_called)
        self.assertTrue(worker.finished.disconnected)

    def test_stop_worker_attrs_keeps_reference_when_worker_is_stuck(self):
        owner = type("Owner", (), {})()
        owner.worker = _Worker(stops=False)
        self.assertEqual(stop_worker_attrs(owner, ("worker",), 10), ["worker"])
        self.assertIsNotNone(owner.worker)


if __name__ == "__main__":
    unittest.main()
