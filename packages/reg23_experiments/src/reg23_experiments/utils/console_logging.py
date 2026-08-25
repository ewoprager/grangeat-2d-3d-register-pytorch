import logging
import re
import shutil
import sys
import textwrap

import tqdm as _tqdm

__all__ = ["ColourFormatter", "tqdm", "indentation_prefix", "TqdmStreamHandler"]

# ANSI escape codes
WHITE = "\033[37m"
RED = "\033[31m"
RED_BACKGROUND = "\033[41m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _len_in_terminal(s: str) -> int:
    return len(ANSI_ESCAPE.sub("", s))


class ColourFormatter(logging.Formatter):
    COLORS = {  #
        logging.DEBUG: WHITE,  #
        logging.INFO: GREEN,  #
        logging.WARNING: YELLOW,  #
        logging.ERROR: RED,  #
        logging.CRITICAL: RED_BACKGROUND,  #
    }

    def format(self, record):
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        level_color = self.COLORS.get(record.levelno, RESET)

        available_width = shutil.get_terminal_size().columns

        prefix = f"[{level_color}{record.levelname:^6s}{RESET}] at [{record.asctime}] "
        suffix0 = f" in [Function '{record.funcName}'] ["
        suffix1 = f"File {record.pathname}:{record.lineno}"
        suffix2 = "]"

        line = prefix + record.message + suffix0 + suffix1 + suffix2
        space = available_width - _len_in_terminal(line)
        if space >= 0:
            return prefix + record.message + ' ' * space + suffix0 + suffix1 + suffix2

        lines = [prefix + record.message]
        if _len_in_terminal(lines[0]) > available_width:
            lines[0] = prefix + ":"
            lines += textwrap.wrap(#
                record.message,#
                width=available_width,#
                initial_indent="└ ",#
                subsequent_indent="    ",#
            )
        lines.append("└ " + suffix0 + suffix1 + suffix2)
        space = available_width - _len_in_terminal(lines[-1])
        if space >= 0:
            lines[-1] = " " * space + "└" + suffix0 + suffix1 + suffix2
        else:
            lines[-1] = "└" + suffix0
            lines.append("    " + suffix1)
            lines.append("  " + suffix2)

        return "\n".join(lines)


def tqdm(*args, **kwargs):
    kwargs.setdefault("disable", not sys.stderr.isatty())
    return _tqdm.tqdm(*args, **kwargs)


def indentation_prefix(tqdm_position: int) -> str:
    if tqdm_position == 0:
        return ""
    return "  " * (tqdm_position - 1) + "└ "


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
