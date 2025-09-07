from __future__ import annotations

from PySide6.QtWidgets import QMenuBar, QAction


class MainMenu(QMenuBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.menu_data = self.addMenu("Data")
        self.menu_model = self.addMenu("Model")
        self.menu_forecast = self.addMenu("Forecast")
        self.menu_tools = self.addMenu("Tools")

        self.action_ingest = QAction("Import/Download", self)
        self.action_resample = QAction("Resample", self)
        self.action_train = QAction("Train", self)
        self.action_eval = QAction("Evaluate", self)
        self.action_forecast = QAction("Forecast", self)
        self.action_realtime = QAction("Realtime ON/OFF", self)
        self.action_calibrator = QAction("Calibrator (ICP)", self)
        self.action_config = QAction("Config", self)
        self.action_toggle_env = QAction("Toggle Envelopes", self)

        self.menu_data.addAction(self.action_ingest)
        self.menu_data.addAction(self.action_resample)
        self.menu_model.addAction(self.action_train)
        self.menu_model.addAction(self.action_eval)
        self.menu_forecast.addAction(self.action_forecast)
        self.menu_tools.addAction(self.action_realtime)
        self.menu_tools.addAction(self.action_calibrator)
        self.menu_tools.addAction(self.action_config)
        self.menu_tools.addAction(self.action_toggle_env)


