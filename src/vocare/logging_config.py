from __future__ import annotations

import logging
import sys

import structlog

from vocare.config import Settings


def configure_logging(settings: Settings) -> None:
    """Structured logs to a local file (for debugging a conversation after the
    fact) plus concise output on stderr, so stdout stays clean for the CLI's
    actual chat transcript."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=settings.vocare_log_level,
    )
    file_handler = logging.FileHandler(settings.vocare_log_file, encoding="utf-8")
    file_handler.setLevel(settings.vocare_log_level)
    logging.getLogger().addHandler(file_handler)

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.vocare_log_level)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
