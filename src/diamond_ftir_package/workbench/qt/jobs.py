"""Run work off the GUI thread; results come back through Qt signals only."""

from __future__ import annotations

import traceback

from qtpy.QtCore import QObject, QRunnable, Signal


class _Signals(QObject):
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)


class Job(QRunnable):
    """``fn(signals)`` runs in the thread pool; its return value arrives in ``done``."""

    def __init__(self, fn):
        super().__init__()
        self.fn, self.signals = fn, _Signals()
        self.setAutoDelete(True)

    def run(self):
        try:
            self.signals.done.emit(self.fn(self.signals))
        except Exception:  # noqa: BLE001 - shown in the GUI
            self.signals.failed.emit(traceback.format_exc(limit=3))
