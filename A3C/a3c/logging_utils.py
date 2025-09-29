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
        except Exception:
            pass


def setup_logger(
    name: str = "a3c",
    level: str = "INFO",
    to_console: bool = True,
    to_file: bool = False,
    log_file: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
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
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(lvl)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if to_file and log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(lvl)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass

    return logger
