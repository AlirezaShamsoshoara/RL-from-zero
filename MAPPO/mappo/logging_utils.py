from __future__ import annotations
import logging
import os
from typing import Optional
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """A logging handler that plays nicely with tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:  # pragma: no cover - defensive
            pass


def setup_logger(
    name: str = "mappo",
    level: str = "INFO",
    to_console: bool = True,
    to_file: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    # Level
    try:
        lvl = getattr(logging, str(level).upper())
    except Exception:
        lvl = logging.INFO
    logger.setLevel(lvl)

    # Formatters
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console via tqdm
    if to_console:
        ch = TqdmLoggingHandler()
        ch.setLevel(lvl)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File
    if to_file and log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(lvl)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception:
            # Fallback to console only if file handler fails
            pass

    return logger
