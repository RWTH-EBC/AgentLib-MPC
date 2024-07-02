"""Some basic plotting utilities"""
import logging
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Callable, TypedDict, Annotated

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator


logger = logging.getLogger(__name__)


@dataclass
class ValueRange:
    min: float
    max: float


Float0to1 = Annotated[float, ValueRange(0.0, 1.0)]
ColorTuple = tuple[Float0to1, Float0to1, Float0to1]


class EBCColors:
    dark_red: ColorTuple = (172 / 255, 43 / 255, 28 / 255)
    red: ColorTuple = (221 / 255, 64 / 255, 45 / 255)
    light_red: ColorTuple = (235 / 255, 140 / 255, 129 / 255)
    green: ColorTuple = (112 / 255, 173 / 255, 71 / 255)
    light_grey: ColorTuple = (217 / 255, 217 / 255, 217 / 255)
    grey: ColorTuple = (157 / 255, 158 / 255, 160 / 255)
    dark_grey: ColorTuple = (78 / 255, 79 / 255, 80 / 255)
    light_blue: ColorTuple = (157 / 255, 195 / 255, 230 / 255)
    blue: ColorTuple = (0 / 255, 84 / 255, 159 / 255)
    ebc_palette_sort_1: list[ColorTuple] = [
        dark_red,
        red,
        light_red,
        dark_grey,
        grey,
        light_grey,
        blue,
        light_blue,
        green,
    ]
    ebc_palette_sort_2: list[ColorTuple] = [
        red,
        blue,
        grey,
        green,
        dark_red,
        dark_grey,
        light_red,
        light_blue,
        light_grey,
    ]


class FontDict(TypedDict):
    fontsize: float


class Style:
    def __init__(self, use_tex: bool = False):
        self.font_dict: FontDict = {"fontsize": 11}
        self.use_tex = use_tex

    def __enter__(self):
        try:
            style_path = Path(Path(__file__).parent, "ebc.paper.mplstyle")
            plt.style.use(style_path)
        except OSError:
            logger.warning("Style Sheet could not be loaded, using default style.")
        if self.use_tex:
            matplotlib.rc("text", usetex=True)
            matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
        # matplotlib.rcParams.update({
        #    "font.family": 'serif',
        #    'font.serif': 'Times',
        # })
        #
        # fontP = FontProperties().set_size('xx-small')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)


Customizer = Callable[[plt.Figure, plt.Axes, Style], Tuple[plt.Figure, plt.Axes]]
MultiCustomizer = Callable[
    [plt.Figure, tuple[plt.Axes], Style], Tuple[plt.Figure, tuple[plt.Axes]]
]


@typing.overload
def make_fig(
    style: Style, customizer: Customizer = None, rows: int = 1
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    ...


@typing.overload
def make_fig(
    style: Style, customizer: MultiCustomizer = None
) -> tuple[plt.Figure, plt.Axes]:
    ...


@typing.overload
def make_fig(style: Style) -> tuple[plt.Figure, plt.Axes]:
    ...


def make_fig(
    style: Style, customizer: Customizer = None, rows=None
) -> Tuple[plt.Figure, tuple[plt.Axes]]:
    """Creates a figure and axes with an amount of rows. If rows is specified, return
    a tuple of axes, else only an ax"""
    if rows is None:
        _rows = 1
    else:
        _rows = rows
    fig, all_ax = plt.subplots(_rows, 1, sharex=True)

    if rows is None:
        # if rows was not specified, return a single axes object
        ax = all_ax
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=style.font_dict["fontsize"],
            left=False,
        )
        if customizer:
            customizer(fig, all_ax, style)
        return fig, all_ax

    # if rows was specified, return a tuple
    if rows == 1:
        all_ax = (all_ax,)

    for ax in all_ax:
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=style.font_dict["fontsize"],
            left=False,
        )
    if customizer:
        customizer(fig, all_ax, style)
    return fig, all_ax


def make_grid(ax: plt.Axes):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(
        which="major",
        axis="both",
        linestyle="--",
        linewidth=0.5,
        color="black",
        zorder=0,
    )
    ax.grid(
        which="minor", axis="both", linestyle="--", linewidth=0.5, color="0.7", zorder=0
    )


def make_side_legend(ax: plt.Axes, fig: plt.Figure = None, right_position: float = 1):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, handlelength=1)
    if fig and right_position > 0:
        fig.subplots_adjust(right=right_position)
