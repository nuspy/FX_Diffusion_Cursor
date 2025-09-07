from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str, app_name: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{app_name}.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    # Rotating file handler
    fh = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    # Reset existing handlers
    root.handlers = []
    root.addHandler(ch)
    root.addHandler(fh)


