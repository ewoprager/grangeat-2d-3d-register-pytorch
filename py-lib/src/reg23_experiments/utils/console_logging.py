import logging
import sys

import tqdm as _tqdm

__all__ = ["tqdm", "TqdmStreamHandler"]


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
