from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    logger_name: str = "crowsnest",
) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    root_logger.handlers.clear()

    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=False,
        show_path=False,
        show_time=True,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(file_handler)

    return logging.getLogger(logger_name)
