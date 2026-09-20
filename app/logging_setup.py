"""Central logging configuration."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from .config import get_settings

_CONFIGURED = False


def configure_logging(level: str | None = None) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("ollama_optimizer")
    if _CONFIGURED:
        if level:
            logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        return logger

    settings = get_settings()
    level = getattr(logging, str(level or settings.log_level).upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    try:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("File logging unavailable; continuing with console logging only.")

    _CONFIGURED = True
    return logger


def get_logger(name: str = "") -> logging.Logger:
    base = configure_logging()
    return base.getChild(name) if name else base
