from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
from PySide6.QtCore import QThread, Signal, QObject

from ..utils.config import load_config, get_default_config_path
from ..utils.logging import setup_logging
from ..services.db import make_engine
from ..services.marketdata import MarketDataService
from ..data.io import backfill_startup
from .menus import MainMenu
from .viewer import TimeSeriesViewer


log = logging.getLogger(__name__)


def _alembic_upgrade(alembic_ini: str) -> None:
    try:
        subprocess.run([sys.executable, "-m", "alembic", "-c", alembic_ini, "upgrade", "head"], check=True)
    except Exception as e:
        log.error("Alembic upgrade failed: %s", e)
        raise


class BackfillWorker(QObject):
    finished = Signal(dict)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self):
        engine = make_engine(self.cfg.app.db_path)
        mds = MarketDataService(base_url=self.cfg.providers.alpha_vantage["base_url"], api_key=self.cfg.providers.alpha_vantage["api_key"], timeout_s=self.cfg.providers.alpha_vantage.get("timeout_s", 30))
        summary = backfill_startup(engine, mds, [i.__dict__ for i in self.cfg.instruments])
        self.finished.emit(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=get_default_config_path())
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(log_dir=cfg.app.data_dir, app_name=cfg.app.name)

    # Run DB migrations at startup
    _alembic_upgrade(cfg.app.alembic_ini)

    app = QApplication(sys.argv)
    win = QMainWindow()
    menu = MainMenu()
    win.setMenuBar(menu)
    viewer = TimeSeriesViewer()
    win.setCentralWidget(viewer)
    win.resize(1024, 640)
    win.setWindowTitle("MagicForex")
    win.show()

    # Async startup backfill
    thread = QThread()
    worker = BackfillWorker(cfg)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    def on_finished(summary):
        # Minimal status feedback in title
        win.setWindowTitle(f"MagicForex - Backfill done {summary}")
        thread.quit()
        thread.wait()

    worker.finished.connect(on_finished)
    thread.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()


