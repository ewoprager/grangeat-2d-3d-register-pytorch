import logging
import pathlib
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import rcParams

__all__ = ["set_mpl_latex_options", "to_latex_scientific", "var_to_string", "latex_escape", "get_colour"]

logger = logging.getLogger(__name__)

MPL_COLOURS = rcParams['axes.prop_cycle'].by_key()['color']


def set_mpl_latex_options():
    # for outputting PGFs
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["scatter.marker"] = 'x'
    plt.rcParams[
        "font.size"] = 11  # figures are includes in latex at quarte size, so 36 is desired size. matplotlib    #
    # scales up by 1.2 (God only knows why). 36 is tool big, however, so going a bit smaller than 30
    rcParams["pgf.texsystem"] = "pdflatex"


def to_latex_scientific(x: float, precision: int = 2, include_plus: bool = False):
    if x == 0:
        return f"{0:.{precision}f}"
    exponent: int = int(f"{x:e}".split("e")[1])
    mantissa: float = x / (10.0 ** exponent)
    if exponent == 0:
        return f"{mantissa:.{precision}f}"
    if include_plus:
        return fr"{mantissa:+.{precision}f} \times 10^{{{exponent}}}"
    return fr"{mantissa:.{precision}f} \times 10^{{{exponent}}}"


def var_to_string(variable_name: str, value: Any) -> str:
    if variable_name == "cropping" or variable_name == "sim_metric":
        return value
    elif variable_name == "mask":
        if value == "None":
            return "no"
        elif value == "Every evaluation weighting zncc":
            return "yes"
        else:
            return value
    elif variable_name == "xray_path":
        return pathlib.Path(value).name
    elif variable_name == "truncation_percent" or variable_name == "downsample_level":
        return f"{value}"
    elif variable_name == "starting_distance":
        return f"{value:.3f}"
    elif variable_name == "crop_expand":
        return f"{value:.1f}"
    try:
        return str(value)
    except Exception:
        return f"<unknown variable '{variable_name}'>"


def get_colour(i):
    return MPL_COLOURS[i % len(MPL_COLOURS)]


def latex_escape(s: str) -> str:
    return (  #
        s.replace("\\", r"\textbackslash{}")  #
        .replace("_", r"\_")  #
        .replace("%", r"\%")  #
        .replace("&", r"\&")  #
        .replace("#", r"\#")  #
        .replace("{", r"\{")  #
        .replace("}", r"\}")  #
    )
