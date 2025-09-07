from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg


class TimeSeriesViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)
        layout.addWidget(self.plot)
        self.candle_curve = None
        self.q05_curve = self.plot.plot(pen=pg.mkPen((150, 150, 255), width=1))
        self.q50_curve = self.plot.plot(pen=pg.mkPen((255, 255, 255), width=2))
        self.q95_curve = self.plot.plot(pen=pg.mkPen((150, 150, 255), width=1))
        self.band = pg.FillBetweenItem(self.q05_curve, self.q95_curve, brush=pg.mkBrush(100, 100, 255, 60))
        self.plot.addItem(self.band)

    def set_data(self, ts_msec: np.ndarray, close: np.ndarray, q05: np.ndarray | None = None, q50: np.ndarray | None = None, q95: np.ndarray | None = None):
        x = ts_msec.astype(np.int64) / 1000.0
        self.plot.clear()
        self.q05_curve = self.plot.plot(x, q05 if q05 is not None else close, pen=pg.mkPen((150, 150, 255), width=1))
        self.q50_curve = self.plot.plot(x, q50 if q50 is not None else close, pen=pg.mkPen((255, 255, 255), width=2))
        self.q95_curve = self.plot.plot(x, q95 if q95 is not None else close, pen=pg.mkPen((150, 150, 255), width=1))
        self.band = pg.FillBetweenItem(self.q05_curve, self.q95_curve, brush=pg.mkBrush(100, 100, 255, 60))
        self.plot.addItem(self.band)


