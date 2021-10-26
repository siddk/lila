"""
overwatch.py

Utility class for creating a centralized/standardized Python logger, with a sane default format, at the appropriate
logging level.
"""
import logging
import sys
from pathlib import Path


# Constants - for Formatting
FORMATTER = logging.Formatter("[*] %(asctime)s - %(name)s - %(levelname)s :: %(message)s", datefmt="%m/%d [%H:%M:%S]")


def get_overwatch(path: Path, run_id: str, level: int, name: str = "lila") -> logging.Logger:
    """
    Initialize logging.Logger with the appropriate name, console, and file handlers.

    :param path: Path for writing log file --> should be identical to run directory (inherited from `train.py`)
    :param run_id: Name of the specific run for writing a custom log file (inherited from `train.py`)
    :param level: Default logging level --> should usually be INFO (inherited from `train.py`).
    :param name: Name of the top-level logger --> should usually be `lila`.

    :return: Default "lila" logger object :: logging.Logger
    """
    # Create Default Logger & add Handlers
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(level)

    # Create Console Handler --> Write to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    logger.addHandler(console_handler)

    # Create File Handler --> Set mode to "w" to overwrite logs (ok, since each run will be uniquely named)
    file_handler = logging.FileHandler(Path(path, f"{run_id}.log"), mode="w")
    file_handler.setFormatter(FORMATTER)
    logger.addHandler(file_handler)

    # Do not propagate by default...
    logger.propagate = False
    return logger
