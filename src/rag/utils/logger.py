import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, debug: bool = False) -> logging.Logger:
    """
    Get a pre-configured logger with console and file handlers.

    Args:
        name (str): Name of the logger, typically __name__.
        debug (bool): If True, set the logging level to DEBUG.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times if the logger is already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)

        # Standard format used across the movie-finder project
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler for persistent logging during ingestion runs
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"rag_ingestion_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.debug(f"Logger initialized for {name} with level {'DEBUG' if debug else 'INFO'}")

    return logger
