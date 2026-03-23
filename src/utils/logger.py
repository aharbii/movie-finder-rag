"""
Centralized logging configuration for the backend application.
"""

import logging
import os
import sys
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOGS_DIR = os.path.join(os.getcwd(), "logs", timestamp)


def get_logger(name: str, debug: bool = False) -> logging.Logger:
    """
    Get a configured logger instance with standard formatting.

    Args:
        name: The name of the logger, typically __name__.
        debug: Enable debug log or not

    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logfile_name = os.path.join(LOGS_DIR, f"{name.replace('.', os.sep)}.log")
        if not os.path.exists(os.path.dirname(logfile_name)):
            os.makedirs(os.path.dirname(logfile_name), exist_ok=True)

        file_handler = logging.FileHandler(logfile_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        full_file_handler = logging.FileHandler(os.path.join(LOGS_DIR, "full.log"))
        full_file_handler.setFormatter(formatter)
        logger.addHandler(full_file_handler)

    return logger
