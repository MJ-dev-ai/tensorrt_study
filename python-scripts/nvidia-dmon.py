#!/usr/bin/env python3
"""nvidia-dmon: NVIDIA GPU monitor with an in-memory store and Start/Stop GUI.

Install:
    python -m pip install PySide6 nvidia-ml-py
Run (GPU index is the NVML/nvidia-smi index):
    python python-scripts/nvidia-dmon.py --gpu 0 --capacity 180 --interval 1
Keep every record until the application closes:
    python python-scripts/nvidia-dmon.py --gpu 0 --capacity -1 --interval 1
Choose a terminal font (default: Cascadia Mono, with installed-font fallbacks):
    python python-scripts/nvidia-dmon.py --font "Cascadia Mono"
GUI demo without an NVIDIA GPU (simulated values, clearly labelled):
    python python-scripts/nvidia-dmon.py --demo

Data flow: Sampler thread -> MemoryDB(deque + lock) -> GUI timer -> chart.
Click Start to collect; Stop pauses collection and keeps the records/graph.
Start again to append to the same history. Pauses appear as gaps in the lines.
--capacity X retains the latest X records. -1 retains all records in RAM.
The graph displays only the latest 100 retained records, regardless of capacity.
You can also edit the limit in the GUI while stopped. Nothing is written to disk.
Unlimited mode uses increasing amounts of RAM as records accumulate.
Closing the window releases NVML and discards all collected records.
sm = time with a GPU kernel executing; mem = VRAM read/write activity time.
Neither is an allocation/occupancy or peak-throughput percentage.

Dependencies: Qt's official PySide6 bindings and NVIDIA's nvidia-ml-py.
The smaller PySide6-Essentials package also suffices for this program.
Sources: https://doc.qt.io/qtforpython-6/
         https://pypi.org/project/nvidia-ml-py/
Package activity checked 2026-10-19:
    PySide6 6.12.0: released 2026-10-08. Official GitHub mirror
    pyside/pyside-setup: 134 stars, updated 2026-10-17;
    primary source repository: https://code.qt.io/pyside/pyside-setup.git/
    nvidia-ml-py 13.615.71: released 2026-09-25; official PyPI
    distribution (no official GitHub star count listed on its project page).
Python 3.10+. NVIDIA driver/NVML required in real monitoring mode.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import math
from itertools import groupby
import signal
import sys
import threading
import time

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:
    raise SystemExit(
        "PySide6 is required. Install with: "
        "python -m pip install PySide6 nvidia-ml-py"
    ) from exc


@dataclass(frozen=True, slots=True)
class Record:
    timestamp: datetime
    elapsed_s: float
    sm: int
    mem: int
    session: int = 0


def terminal_font(preferred: str = "Cascadia Mono") -> QtGui.QFont:
    """Match Windows Terminal with a regular 12-point antialiased monospace font."""
    installed = {family.casefold(): family for family in QtGui.QFontDatabase.families()}
    candidates = (preferred, "Cascadia Mono", "Cascadia Code", "Consolas",
                  "DejaVu Sans Mono", "Liberation Mono")
    family = next((installed[name.casefold()] for name in candidates
                   if name.casefold() in installed), None)
    if family is None:
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
    else:
        font = QtGui.QFont(family, 12)
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    font.setPointSize(12)
    font.setWeight(QtGui.QFont.Weight.Normal)
    font.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
    return font


class MemoryDB:
    """One writer, any number of readers; snapshots cannot mutate stored rows."""

    def __init__(self, capacity: int = 180):
        self._validate_capacity(capacity)
        self.capacity = capacity
        self._rows: deque[Record] = deque(maxlen=None if capacity == -1 else capacity)
        self._lock = threading.Lock()
        self._total = 0

    @staticmethod
    def _validate_capacity(capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or (capacity != -1 and capacity < 1):
            raise ValueError("capacity must be -1 (unlimited) or a positive integer")

    def set_capacity(self, capacity: int) -> None:
        """Resize while stopped; reducing the limit keeps the newest records."""
        self._validate_capacity(capacity)
        with self._lock:
            self._rows = deque(self._rows, maxlen=None if capacity == -1 else capacity)
            self.capacity = capacity

    def insert(self, record: Record) -> None:
        with self._lock:
            # At capacity, append automatically evicts the oldest record.
            self._rows.append(record)
            self._total += 1

    def snapshot(self, after_total: int | None = None) -> tuple[int, tuple[Record, ...]] | None:
        with self._lock:
            if after_total == self._total:
                return None  # Avoid copying unlimited history on every GUI poll.
            return self._total, tuple(self._rows)


class NvmlSource:
    """Construct, query and close in the sampler thread only."""

    def __init__(self, gpu: int):
        try:
            import pynvml  # Install nvidia-ml-py, not the deprecated pynvml package.
        except ImportError as exc:
            raise RuntimeError("Install NVIDIA bindings: python -m pip install nvidia-ml-py") from exc
        self.nvml = pynvml
        pynvml.nvmlInit()
        try:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            self.name = f"GPU {gpu}: {name}"
        except Exception:
            pynvml.nvmlShutdown()
            raise

    def read(self) -> tuple[int, int]:
        util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
        sm, mem = int(util.gpu), int(util.memory)
        if not (0 <= sm <= 100 and 0 <= mem <= 100):
            raise RuntimeError("This device/driver did not return supported utilization values.")
        return sm, mem

    def close(self) -> None:
        self.nvml.nvmlShutdown()


class DemoSource:
    name = "DEMO - simulated values (not GPU measurements)"

    def __init__(self, _gpu: int):
        self._start = time.monotonic()

    def read(self) -> tuple[int, int]:
        t = time.monotonic() - self._start
        return round(35 + 25 * math.sin(t / 4)), round(20 + 15 * math.sin(t / 3 + 1))

    def close(self) -> None:
        pass


class Sampler(QtCore.QThread):
    ready = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, db: MemoryDB, gpu: int, interval: float, demo: bool,
                 origin: float | None = None, session: int = 0, parent=None):
        super().__init__(parent)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        self.db = db
        self.gpu = gpu
        self.interval = interval
        self.demo = demo
        self.origin = time.monotonic() if origin is None else origin
        self.session = session
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()  # Interrupts the wait between samples immediately.

    def run(self) -> None:
        source = None
        try:
            source = (DemoSource if self.demo else NvmlSource)(self.gpu)
            self.ready.emit(source.name)
            deadline = time.monotonic()
            while not self._stop_event.wait(max(0.0, deadline - time.monotonic())):
                sm, mem = source.read()
                self.db.insert(Record(datetime.now().astimezone(), time.monotonic() - self.origin, sm, mem, self.session))
                deadline += self.interval
                now = time.monotonic()
                if deadline <= now:
                    # Skip missed ticks instead of inserting a burst of fake history.
                    deadline = now + self.interval
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as exc:
                    self.failed.emit(f"NVML shutdown: {exc}")


class HistoryChart(QtWidgets.QWidget):
    """Qt Widgets/QPainter chart; no CUDA, OpenGL or extra plotting package."""

    COLORS = ("#8cace6", "#ddb58c")
    DISPLAY_LIMIT = 100

    def __init__(self, capacity: int, interval: float):
        super().__init__()
        self.set_capacity(capacity, interval)
        self.rows: tuple[Record, ...] = ()
        self.setMinimumSize(520, 280)

    def set_capacity(self, capacity: int, interval: float) -> None:
        visible_capacity = self.DISPLAY_LIMIT if capacity == -1 else min(capacity, self.DISPLAY_LIMIT)
        self.window_s = max(1.0, interval * max(1, visible_capacity - 1))
        self.update()

    def set_records(self, rows: tuple[Record, ...]) -> None:
        self.rows = rows[-self.DISPLAY_LIMIT:]
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#0c0c0c"))
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        plot = QtCore.QRectF(52, 24, self.width() - 76, self.height() - 68)
        latest = self.rows[-1].elapsed_s if self.rows else 0
        right = max(self.window_s, latest)
        left = right - self.window_s
        if self.rows:
            left = min(left, self.rows[0].elapsed_s)  # Keep pre-pause records visible.
        span = right - left
        align = QtCore.Qt.AlignmentFlag
        font = self.font()
        font.setPointSize(12)
        painter.setFont(font)
        for value in range(0, 101, 20):
            y = plot.bottom() - plot.height() * value / 100
            painter.setPen(QtGui.QColor(255, 255, 255, 40))
            painter.drawLine(QtCore.QPointF(plot.left(), y), QtCore.QPointF(plot.right(), y))
            painter.setPen(QtGui.QColor("#cccccc"))
            painter.drawText(QtCore.QRectF(0, y - 9, 44, 18), align.AlignRight | align.AlignVCenter, str(value))
        for tick in range(5):
            elapsed = left + span * tick / 4
            x = plot.left() + plot.width() * tick / 4
            precision = 1 if span < 10 else 0
            painter.drawText(QtCore.QRectF(x - 36, plot.bottom() + 8, 72, 18), align.AlignCenter, f"{elapsed:.{precision}f}")
        painter.drawText(QtCore.QRectF(4, 2, 48, 18), align.AlignCenter, "%")
        painter.drawText(QtCore.QRectF(plot.left(), self.height() - 22, plot.width(), 18), align.AlignCenter, "Elapsed time (s)")
        for offset, name, color in ((180, "SM", self.COLORS[0]), (88, "MEM", self.COLORS[1])):
            x = plot.right() - offset
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 2))
            painter.drawLine(QtCore.QPointF(x, 10), QtCore.QPointF(x + 22, 10))
            painter.setPen(QtGui.QColor("#cccccc"))
            painter.drawText(QtCore.QRectF(x + 28, 0, 60, 20), align.AlignVCenter, name)
        painter.setClipRect(plot.adjusted(-2, -2, 2, 2))
        for field, color in zip(("sm", "mem"), self.COLORS):
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 2))
            for _, session_rows in groupby(self.rows, key=lambda row: row.session):
                points = QtGui.QPolygonF()
                pixel = lambda row: int(plot.width() * (row.elapsed_s - left) / span)
                for _, bucket in groupby(session_rows, key=pixel):
                    samples = list(bucket)
                    # Preserve peaks while limiting painted vertices per pixel.
                    indices = sorted({0, len(samples) - 1,
                                      min(range(len(samples)), key=lambda i: getattr(samples[i], field)),
                                      max(range(len(samples)), key=lambda i: getattr(samples[i], field))})
                    for index in indices:
                        row = samples[index]
                        points.append(QtCore.QPointF(
                            plot.left() + plot.width() * (row.elapsed_s - left) / span,
                            plot.bottom() - plot.height() * getattr(row, field) / 100,
                        ))
                if len(points) > 1:
                    painter.drawPolyline(points)
                if points:
                    painter.setBrush(QtGui.QColor(color))
                    painter.drawEllipse(points[-1], 2, 2)
        painter.end()


class MonitorWindow(QtWidgets.QWidget):
    def __init__(self, capacity: int, interval: float, gpu: int, demo: bool = False,
                 font_family: str = "Cascadia Mono"):
        super().__init__()
        self.db = MemoryDB(capacity)
        self.gpu, self.interval, self.demo = gpu, interval, demo
        self.sampler: Sampler | None = None
        self._origin: float | None = None
        self._session = 0
        self._seen_total = -1
        self._closing = False
        self._failed = False
        self.setWindowTitle("nvidia-dmon" + (" [DEMO]" if demo else ""))
        self.resize(1020, 560)
        self.setFont(terminal_font(font_family))
        self.setStyleSheet("""
            QWidget { background-color: #0c0c0c; color: #cccccc; }
            QLineEdit { border: 1px solid #767676; padding: 5px;
                        selection-background-color: #cccccc; selection-color: #0c0c0c; }
            QPushButton { border: 1px solid #767676; padding: 8px 20px; }
            QPushButton:pressed { background-color: #222222; }
            QPushButton:disabled, QLineEdit:disabled { border-style: dotted; }
        """)
        layout = QtWidgets.QVBoxLayout(self)
        self.device = QtWidgets.QLabel("nvidia-dmon" + (" [DEMO - SIMULATED DATA]" if demo else ""))
        title_font = QtGui.QFont(self.font())
        title_font.setPointSize(12)
        title_font.setBold(False)
        self.device.setFont(title_font)
        layout.addWidget(self.device)
        self.sm_stats = QtWidgets.QLabel("SM activity: waiting for the first sample")
        self.mem_stats = QtWidgets.QLabel("VRAM read/write activity: waiting for the first sample")
        stats_font = QtGui.QFont(self.font())
        stats_font.setPointSize(12)
        self.sm_stats.setFont(stats_font)
        self.mem_stats.setFont(stats_font)
        layout.addWidget(self.sm_stats)
        layout.addWidget(self.mem_stats)
        self.chart = HistoryChart(capacity, interval)
        layout.addWidget(self.chart, 1)
        self.count = QtWidgets.QLabel(f"Retained 0/{'ALL' if capacity == -1 else capacity} | GPU {gpu} | interval {interval:g} s | RAM only")
        layout.addWidget(self.count)
        self.status = QtWidgets.QLabel("Ready. Click Start. Max records: -1 = keep all until exit.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Max records:"))
        self.capacity_edit = QtWidgets.QLineEdit(str(capacity))
        self.capacity_edit.setMaximumWidth(110)
        self.capacity_edit.setValidator(QtGui.QIntValidator(-1, 2147483647, self))
        controls.addWidget(self.capacity_edit)
        controls.addWidget(QtWidgets.QLabel("(-1 = ALL)"))
        controls.addStretch()
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self._start)
        controls.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(200)  # GUI polling; sampling still occurs only every interval.
        self.timer.timeout.connect(self._refresh)
        self.timer.start()

    def _start(self) -> None:
        if self.sampler is not None or self._closing:
            return
        try:
            capacity = int(self.capacity_edit.text())
            if capacity != -1 and not math.isfinite(capacity * self.interval):
                raise ValueError("capacity * interval is too large")
            self.db.set_capacity(capacity)
        except ValueError as exc:
            self.status.setText(f"Invalid record limit: {exc}")
            return
        self.chart.set_capacity(capacity, self.interval)
        self._seen_total = -1
        self._failed = False
        if self._origin is None:
            self._origin = time.monotonic()
        self._session += 1
        self.sampler = Sampler(self.db, self.gpu, self.interval, self.demo,
                               self._origin, self._session, self)
        self.sampler.ready.connect(self._ready)
        self.sampler.failed.connect(self._error)
        self.sampler.finished.connect(self._finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.capacity_edit.setEnabled(False)
        self.status.setText("Starting sampler...")
        self.sampler.start()
        self._refresh()

    def _ready(self, name: str) -> None:
        self.device.setText(f"nvidia-dmon | {name}")
        if self.stop_button.isEnabled():
            self.status.setText("Collecting. Mean/peak cover retained records. MEM = VRAM activity, not allocated VRAM.")

    def _refresh(self) -> None:
        snapshot = self.db.snapshot(self._seen_total)
        if snapshot is None:
            return
        total, rows = snapshot
        self._seen_total = total
        self.chart.set_records(rows)
        limit = "ALL" if self.db.capacity == -1 else str(self.db.capacity)
        if not rows:
            self.count.setText(f"Retained 0/{limit} | GPU {self.gpu} | interval {self.interval:g} s | RAM only")
            return
        for label, field, title in ((self.sm_stats, "sm", "SM activity"), (self.mem_stats, "mem", "VRAM activity")):
            values = [getattr(row, field) for row in rows]
            label.setText(f"{title}: {values[-1]}%    |    Mean {sum(values) / len(values):.1f}%    |    Peak {max(values)}%")
        span = rows[-1].elapsed_s - rows[0].elapsed_s
        self.count.setText(f"Retained {len(rows)}/{limit} | Chart {len(self.chart.rows)}/100 | Total {total} | History {span:.1f} s | Latest {rows[-1].timestamp:%H:%M:%S} | RAM only")

    def _stop(self) -> None:
        if self.sampler is None:
            return
        self.sampler.request_stop()
        self.stop_button.setEnabled(False)
        self.status.setText("Stopping sampler...")

    def _error(self, message: str) -> None:
        self._failed = True
        self.status.setText(f"Monitoring stopped: {message}")
        print(message, file=sys.stderr)

    def _finished(self) -> None:
        self._refresh()
        sampler = self.sampler
        self.sampler = None
        if sampler is not None:
            sampler.deleteLater()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(not self._closing)
        self.capacity_edit.setEnabled(not self._closing)
        if not self._failed:
            self.status.setText("Stopped. Records retained. Click Start to append to the same history.")
        if self._closing:
            QtCore.QTimer.singleShot(0, self.close)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.sampler is not None and self.sampler.isRunning():
            # Do not destroy a running QThread or block the GUI with wait().
            self._closing = True
            self._stop()
            event.ignore()
        else:
            self.timer.stop()
            event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(prog="nvidia-dmon", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0, help="NVML/nvidia-smi GPU index (default: 0)")
    parser.add_argument("--capacity", type=int, default=180, help="Maximum retained records; -1 = unlimited (default: 180)")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1)")
    parser.add_argument("--demo", action="store_true", help="Explicit simulation; does not access NVIDIA/NVML")
    parser.add_argument("--font", default="Cascadia Mono", help="Preferred installed font (default: Cascadia Mono)")
    args = parser.parse_args()
    if (args.capacity != -1 and args.capacity < 1) or args.gpu < 0 or not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("capacity = -1 or >= 1, gpu >= 0, and a positive finite interval are required")
    if args.capacity != -1 and not math.isfinite(args.interval * args.capacity):
        parser.error("capacity * interval is too large")
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("nvidia-dmon")
    app.setApplicationDisplayName("nvidia-dmon")
    app.setStyle("Fusion")
    window = MonitorWindow(args.capacity, args.interval, args.gpu, args.demo, args.font)
    signal.signal(signal.SIGINT, lambda *_: window.close())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
