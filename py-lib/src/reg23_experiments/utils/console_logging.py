import logging
import sys

import tqdm as _tqdm

__all__ = ["ColourFormatter", "tqdm", "TqdmStreamHandler"]

# ANSI escape codes
WHITE = "\033[37m"
RED = "\033[31m"
RED_BACKGROUND = "\033[41m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class ColourFormatter(logging.Formatter):
    COLORS = {  #
        logging.DEBUG: WHITE,  #
        logging.INFO: GREEN,  #
        logging.WARNING: YELLOW,  #
        logging.ERROR: RED,  #
        logging.CRITICAL: RED_BACKGROUND,  #
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, RESET)
        record.levelname = f"{color}{record.levelname}{RESET}"
        return super().format(record)


def tqdm(*args, **kwargs):
    kwargs.setdefault("disable", not sys.stderr.isatty())
    return _tqdm.tqdm(*args, **kwargs)


class TqdmStreamHandler(logging.StreamHandler):
    """
    A stream handler that writes to the console in a way compatible with tqdm progress bars
    """

    def emit(self, record):
        try:
            msg = self.format(record)
            _tqdm.tqdm.write(msg)  # This writes safely above a tqdm progress bar
            self.flush()
        except Exception:
            self.handleError(record)
