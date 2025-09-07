from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QLabel

from ..utils.config import load_config, get_default_config_path
from ..utils.logging import setup_logging


log = logging.getLogger(__name__)


def _alembic_upgrade(alembic_ini: str) -> None:
    try:
        subprocess.run([sys.executable, "-m", "alembic", "-c", alembic_ini, "upgrade", "head"], check=True)
    except Exception as e:
        log.error("Alembic upgrade failed: %s", e)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=get_default_config_path())
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(log_dir=cfg.app.data_dir, app_name=cfg.app.name)

    # Run DB migrations at startup
    _alembic_upgrade(cfg.app.alembic_ini)

    app = QApplication(sys.argv)
    # Placeholder GUI; real viewer/menus will be added next steps
    w = QLabel("MagicForex GUI - placeholder. Use menus in next iteration.")
    w.resize(480, 120)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


