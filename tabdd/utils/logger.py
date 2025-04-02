import logging
from datetime import datetime

from tabdd.config.paths import LOGGER_DIR



def setup_logger():
    """Setup and return a logger with specified name and level, including file handler."""
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.INFO)
    # Prevent log messages from being duplicated in the Python root logger
    logger.propagate = False
    # Check if handlers are already added (important in scenarios where setup_logger might be called multiple times)
    if not logger.handlers:
        # Create a console handler and set the level to info
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.terminator = "\n"

        # Create a file handler and set the level to info
        # cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # log_file = LOGGER_DIR / f"{cur_time}-experiment.log"
        # fh = logging.FileHandler(log_file)
        # fh.setLevel(logging.INFO)
        # fh.terminator = "\n"

        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            "%(asctime)s - %(message)s"
        )
        ch.setFormatter(formatter)
        # fh.setFormatter(formatter)

        logger.addHandler(ch)
        # logger.addHandler(fh)

    return logger
