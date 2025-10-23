from __future__ import annotations

import logging
import os
from typing import Optional

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Logging handler compatible with tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:  # pragma: no cover - defensive
            pass


def setup_logger(
    name: str = "nash-ql",
    level: str = "INFO",
    to_console: bool = True,
    to_file: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logger with tqdm-compatible console handler and optional file handler.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        to_console: Whether to log to console
        to_file: Whether to log to file
        log_file: Path to log file (required if to_file=True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    try:
        lvl = getattr(logging, str(level).upper())
    except Exception:
        lvl = logging.INFO
    logger.setLevel(lvl)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if to_console:
        ch = TqdmLoggingHandler()
        ch.setLevel(lvl)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if to_file and log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(lvl)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception:
            pass

    return logger
