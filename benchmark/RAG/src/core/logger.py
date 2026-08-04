import logging
import os


class _ConsoleInfoFilter(logging.Filter):
    """Keep console output readable while preserving full logs in the file."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return record.name == "Benchmark"


def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("Benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.addFilter(_ConsoleInfoFilter())
    logger.addHandler(sh)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("Benchmark")
