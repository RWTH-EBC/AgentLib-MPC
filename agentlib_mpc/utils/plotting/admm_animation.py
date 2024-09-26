"""Modules that defines functions to be used for automatically creating animations of
ADMM convergence"""

import functools
from pathlib import Path
from typing import Iterable, Union, Callable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

from agentlib_mpc.utils.analysis import (
    admm_at_time_step,
    load_admm,
    get_number_of_iterations,
)
from agentlib_mpc.utils.plotting.basic import make_grid, make_fig, Style, Customizer

Label = str
LinesDict = dict[Label, plt.Line2D]
Data = dict[Label, pd.DataFrame]
Init = Callable[[], None]
Animate = Callable[[Union[int, Iterable]], None]


def make_lines(labels: list[Label], ax: plt.Axes, fig: plt.Figure) -> LinesDict:
    lines: LinesDict = {}

    for label in labels:
        lines[label] = ax.plot([], [], lw=2, label=str(label))[0]

    return lines


def init_full(lines: LinesDict, annotation: plt.Annotation, ax: plt.Axes):
    for line in lines.values():
        line.set_data([], [])

    ax.legend(list(lines.values()), list(lines))
    # annotation
    return tuple(lines.values()) + (annotation,)


def animate_full(
    i: int, lines: LinesDict, annotation: plt.Annotation, data: Data, time_step: float
):  # Upper plot: Temperatures
    for label, line in lines.items():
        data_for_iter = admm_at_time_step(
            data=data[label],
            time_step=time_step,
            iteration=i,
        ).dropna()

        line.set_data(data_for_iter.index, data_for_iter)

    # annotation
    annotation.set_text(f"Iteration: {i}")

    print(f"Made Frame {i}")

    return tuple(lines.values()) + (annotation,)


def make_image(
    data: Data,
    time_step: float = 0,
    file_name: str = "",
    customize: Customizer = None,
    iteration=-1,
):
    labels = list(data)  # from data

    with Style() as style:
        fig, ax = make_fig(style)
        if customize:
            fig, ax = customize(fig, ax)
        lines = make_lines(labels, ax=ax, fig=fig)
        annotation = ax.annotate(
            xy=(0.1, 0.1),
            xytext=(0.5, 1.05),
            text="Iteration: 0",
            animated=True,
            textcoords="axes fraction",
            xycoords="axes fraction",
            ha="center",
        )

        animate = functools.partial(
            animate_full,
            annotation=annotation,
            lines=lines,
            data=data,
            time_step=time_step,
        )
        init = functools.partial(init_full, annotation=annotation, lines=lines, ax=ax)
        init()
        animate(i=iteration)
        if file_name:
            fig.savefig(fname=file_name)


def make_animation(
    data: Data,
    time_step: float = 0,
    file_name: str = "",
    customize: Customizer = None,
    iteration=-1,
    interval: int = 300,
):
    labels = list(data)  # from data

    with Style() as style:
        fig, ax = make_fig(style)
        if customize:
            fig, ax = customize(fig, ax)
        lines = make_lines(labels, ax=ax, fig=fig)
        annotation = ax.annotate(
            xy=(0.1, 0.1),
            xytext=(0.5, 1.05),
            text="Iteration: 0",
            animated=True,
            textcoords="axes fraction",
            xycoords="axes fraction",
            ha="center",
        )

        animate = functools.partial(
            animate_full,
            annotation=annotation,
            lines=lines,
            data=data,
            time_step=time_step,
        )
        init = functools.partial(init_full, annotation=annotation, lines=lines, ax=ax)

    # setup_figure()
    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=iteration,
        interval=interval,
        blit=True,
        repeat_delay=1500,
    )
    if not file_name.endswith(".gif"):
        raise ValueError(
            f"Target filename needs '.gif' extension. Given filename was {file_name}"
        )
    anim.save(file_name, writer="imagemagick")


if __name__ == "__main__":

    def customize_fig(fig: plt.Figure, ax: plt.Axes) -> (plt.Figure, plt.Axes):
        # grids
        make_grid(ax)

        # auxiliary
        ax.set_ylim(0, 0.11)
        ax.set_xlim(0, 3000)
        ax.legend()
        # cax.get_legend().remove()
        ax.set_ylabel("Temperature / $°C$")

        # ticks
        # xticks = np.arange(mpc_log.index[0], mpc_log.index[-1] + 1, 24 * 3600)
        # xtickval = [str(i) for i, _ in enumerate(xticks)]
        # plt.xticks(xticks, xtickval)
        ax.set_xlabel("Time / h")
        return fig, ax

    test_dir = Path(
        r"C:\Users\ses\Dokumente\Vorträge\Freitagsvorträge\2021September\admm"
    )
    filename = "first_anim.gif"
    room = load_admm(Path(test_dir, "admm_opt.csv"))
    cooler = load_admm(Path(test_dir, "cooler_res.csv"))
    data_ = {
        Label("room_T"): room["variable"]["mDot_0"],
        Label("cooler_T"): cooler["variable"]["mDot"],
    }

    iter_dict = get_number_of_iterations(room)
    iters = pd.Series(iter_dict).iloc[0]

    make_animation(
        file_name=filename,
        data=data_,
        customize=customize_fig,
        time_step=500,
        iteration=iters,
    )
