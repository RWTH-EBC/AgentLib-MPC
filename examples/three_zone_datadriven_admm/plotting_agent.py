import os
import json
import time
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict


dark_red = [172 / 255, 43 / 255, 28 / 255]
red = [221 / 255, 64 / 255, 45 / 255]
light_red = [235 / 255, 140 / 255, 129 / 255]
green = [112 / 255, 173 / 255, 71 / 255]
light_grey = [217 / 255, 217 / 255, 217 / 255]
grey = [157 / 255, 158 / 255, 160 / 255]
dark_grey = [78 / 255, 79 / 255, 80 / 255]
light_blue = [157 / 255, 195 / 255, 230 / 255]
blue = [0 / 255, 84 / 255, 159 / 255]
colors = {
    "dark_red": dark_red,
    "red": red,
    "light_red": light_red,
    "green": green,
    "light_grey": light_grey,
    "grey": grey,
    "dark_grey": dark_grey,
    "light_blue": light_blue,
    "blue": blue,
}
color_list = [
    red,
    blue,
    green,
    grey,
    dark_red,
    dark_grey,
    light_blue,
    light_red,
    light_grey,
]


class Plotter:
    def __init__(self):
        pass

    def plot_zones(
        self,
        ax,
        data: List[pd.DataFrame],
        plotted_var: List[str],
        colors_string: List[str],
        labels: List[str],
        celsius: bool = False,
        step: bool = False,
        startingindex: int = 0,
    ):
        """
        creates a figure and axis, plotting all data specified in the input
        and returns a list of labels for the legend.
        Each subplot plots the variable trajectories of one zone.
        e.g. subplot1 for zone-temperature, disturbances and CCA-temperature of zone 1
        """
        if len(data) > ax.shape[0]:
            raise ValueError("More data available than the subplots can handle")
        # plot temperatures given in data
        lns = []
        index = data[0].index / 3600
        for v in range(len(plotted_var)):
            for i in range(len(data)):
                if not step:
                    lns1 = ax[i + startingindex].plot(
                        index,
                        data[i][plotted_var[v]] - celsius * 273.15,
                        label=labels[v],
                        color=colors[colors_string[v]],
                    )
                else:
                    lns1 = ax[i + startingindex].step(
                        index,
                        data[i][plotted_var[v]] - celsius * 273.15,
                        label=labels[v],
                        color=colors[colors_string[v]],
                        where="post",
                        ls="--",
                    )
                if i == (len(data) - 1):
                    lns.append(lns1)

        for i in range(len(data)):
            ax[i].grid()
            ax[i].title.set_text(f"Zone {i+1}")

        return lns

    def plot_variables(
        self,
        ax,
        data: List[pd.DataFrame],
        plotted_var: List[str],
        labels: List[str],
        celsius: bool = False,
        step: bool = False,
        startingindex: int = 0,
    ):
        """
        creates a figure and axis, plotting all data specified in the input
        and returns a list of labels for the legend.
        Each subplot plots the variable trajectories of all zones of one variable.
        e.g. subplot1 for the energy consumption in zone 1,2,3 etc.
        """
        if len(data) > ax.shape[0]:
            raise ValueError("More data available than the subplots can handle")
        # plot temperatures given in data
        lns = []
        index = data[0].index / 3600
        # for v in range(len(plotted_var)):
        #    for i in range(len(data)):
        for i in range(len(data)):
            for v in range(len(plotted_var)):
                if not step:
                    lns1 = ax[v + startingindex].plot(
                        index,
                        data[i][plotted_var[v]] - celsius * 273.15,
                        label=labels[i],
                        color=colors[i],
                    )
                else:
                    lns1 = ax[v + startingindex].step(
                        index,
                        data[i][plotted_var[v]] - celsius * 273.15,
                        label=labels[i],
                        color=colors[i],
                        where="post",
                        ls="--",
                    )
                if v == (len(plotted_var) - 1):
                    lns.append(lns1)

        return lns

    def plot_rightaxis(
        self,
        ax,
        data: List[pd.DataFrame],
        plotted_var: List[str],
        labels: List[str],
        sumvar: bool = True,
    ):
        axright = []
        for i in range(ax.shape[0] - 1):
            axright.append(ax[i].twinx())
        index = data[0].index / 3600
        lns = []
        if sumvar:
            for i in range(len(data)):
                sum_variables = 0
                # calculate the sum
                for var in plotted_var:
                    sum_variables += data[i][var]
                # use plot or step
                lns1 = ax[i].plot(
                    index, sum_variables, label=labels[0], color="black", ls=":"
                )
            lns.append(lns1)
        axright[1].set_ylabel("Disturbance [W]")
        for i in range(ax.shape[0] - 1):
            axright[i].set_ylim([-100, 600])
        return lns

    def plot_subplot(
        self,
        ax,
        data,
        plotted_var=None,
        labels: List[str] = None,
        celsius: bool = False,
        step: bool = False,
        startingindex: int = 0,
        index=None,
        y_lim=None,
    ):
        if len(data) != len(labels):
            raise ValueError("numbers of data and labels dont match")
        # plot temperatures given in data
        if plotted_var is not None:
            index = data[0].index / 3600
            for v in range(len(data)):
                if not step:
                    ax[startingindex].plot(
                        index,
                        data[v][plotted_var[v]] - celsius * 273.15,
                        label=labels[v],
                        color=color_list[v],
                    )
                else:
                    ax[startingindex].step(
                        index,
                        data[v][plotted_var[v]] - celsius * 273.15,
                        label=labels[v],
                        color=color_list[v],
                        where="post",
                    )
        else:
            if index is not None:
                for v in range(len(labels)):
                    if not step:
                        ax[startingindex].plot(
                            index,
                            np.array(data[v]) - celsius * 273.15,
                            label=labels[v],
                            color=color_list[v],
                        )
                    else:
                        ax[startingindex].step(
                            index,
                            np.array(data[v]) - celsius * 273.15,
                            label=labels[v],
                            color=color_list[v],
                            where="post",
                        )
            else:
                for v in range(len(labels)):
                    if not step:
                        ax[startingindex].plot(
                            np.array(data[v]) - celsius * 273.15,
                            label=labels[v],
                            color=color_list[v],
                        )
                    else:
                        ax[startingindex].step(
                            np.array(data[v]) - celsius * 273.15,
                            label=labels[v],
                            color=color_list[v],
                            where="post",
                        )
        if y_lim is not None:
            ax[startingindex].set_ylim(y_lim)

    def plot_admm_dev(self, admm_residuals):
        plt.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "font.size": 9,
                "text.usetex": True,
                "svg.fonttype": "none",
                "pgf.rcfonts": False,
            }
        )
        fig7, ax7 = plt.subplots(1, 1)
        admm_res = np.array(admm_residuals)
        primal_res = admm_res[:, 3]
        dual_res = admm_res[:, 4]
        rho = admm_res[:, 5]
        ln1 = ax7.plot(
            primal_res,
            label=f"Primal residual",
            marker="o",
            markerfacecolor="None",
            color=red,
            linestyle="None",
            markersize=3,
        )
        ln2 = ax7.plot(
            dual_res,
            label=f"Dual residual",
            marker="o",
            markerfacecolor="None",
            color=blue,
            linestyle="None",
            markersize=3,
        )
        ax_right1 = ax7.twinx()
        ln3 = ax_right1.plot(
            rho,
            label=f"Penalty factor",
            marker="x",
            color=green,
            markersize=1,
            linestyle="None",
        )
        ax7.set_ylabel("Residual")
        ax_right1.set_ylabel("Penalty factor")
        for i in range(len(admm_res)):
            if admm_res[i, 2] == 0:
                ax7.axvline(i, color=grey, ls="--")
        lns = ln1 + ln2 + ln3
        labs = [lab.get_label() for lab in lns]
        ax7.legend(
            lns,
            labs,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            fancybox=True,
            shadow=True,
            ncol=3,
        )
        ax7.set_yscale("log")
        fig7.set_size_inches(w=6, h=3)
        ax7.grid()

    def plotdR(self, res_sim):
        fig6, ax6 = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"wspace": 0, "hspace": 0.07},
            constrained_layout=True,
        )
        ax6[0].plot(res_sim["T_0_out"], label="$T_{Zone}$", color=red)
        ax6[0].plot(res_sim["T_CCA_out"], label="$T_{CCA}$", color=blue)
        ax_right1 = ax6[0].twinx()
        ax_right1.plot(
            res_sim["d"] + res_sim["Q_rad"], label="Disturbance", color="black", ls=":"
        )
        ax6[0].step(
            res_sim.index,
            res_sim["T_upper"],
            color=grey,
            where="post",
            label="$T_{bound}$",
            ls="--",
        )
        ax6[0].step(
            res_sim.index, res_sim["T_lower"], color=grey, where="post", ls="--"
        )
        vorlauftemp1 = res_sim["T_vorlauf"]
        ahutemp1 = res_sim["T_ahu"]
        ax6[1].plot(vorlauftemp1, label="$T_{sup}$", color=red)
        ax6[1].plot(ahutemp1, label="$T_{ahu}$ Z1", color=blue, ls="-.")
        fig6.set_size_inches(w=6.1, h=4)
        fig6.tight_layout(pad=3.0)
        ax_right1.set_ylabel("Disturbance [W]")
        ax6[0].set_ylabel("Temperature [K]")
        ax6[1].set_ylabel("Temperature [K]")
        ax6[0].title.set_text("Zone 1")
        ax6[1].title.set_text("Control temperature")
        ax6[1].set_ylim([280, 310])
        ax6[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            fancybox=True,
            shadow=True,
            ncol=5,
        )
        ax6[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            fancybox=True,
            shadow=True,
            ncol=5,
        )
        ax6[0].grid()
        ax6[1].grid()
        fig6.tight_layout()
