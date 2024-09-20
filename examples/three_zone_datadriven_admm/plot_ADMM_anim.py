from pathlib import Path

from agentlib_mpc.utils.analysis import admm_at_time_step, load_admm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import pandas as pd


############################## PARAMETERS ##################################

time_step = 0
filename = "agreement_at_10.png"
file_format = "eps"

################################ STYLE #######################################

# plt.style.use(r'C:\Users\ses\repos\matplolib-style\ebc.paper.mplstyle')
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
# matplotlib.rcParams.update({
#     "font.family": 'serif',
#     'font.serif': 'Times',
# })
#
fontdict = {"fontsize": 11}
# fontP = FontProperties().set_size('xx-small')


############################ LOAD DATA #######################################

# mpc_log = AgentLogger.load_filename_log(filename='Agent_Simulation_Logger.log',
#                                         values_only=True)

# room_results1, _ = MPC.read_results_file(results_file='admm_opt.csv',clean_up=False)
# room_results1 = room_results1.drop(labels='time_step',axis=0)
# room_results2, _ = MPC.read_results_file(results_file='admm_opt2.csv',clean_up=False)
# room_results2 = room_results2.drop(labels='time_step',axis=0)
# cooler_results, _ = MPC.read_results_file(results_file='tempcontroller_res.csv',clean_up=False)
# cooler_results = cooler_results.drop(labels='time_step',axis=0)
room_results1 = load_admm(Path("admm_opt.csv"))
room_results2 = load_admm(Path("admm_opt2.csv"))
room_results3 = load_admm(Path("admm_opt3.csv"))
cooler_results = load_admm(Path("tempcontroller_res.csv"))

############################# SETUP FIGURE ####################################
fig, ax = plt.subplots(1, 1)


def setup_figure():
    cm = 1 / 2.54
    fig.set_size_inches(15 * cm, 9 * cm)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fontdict["fontsize"],
        left=False,
    )


########################### SETUP ANIMATION ##################################
(line_room,) = ax.plot([], [], lw=2, label="Room 1")
(line_room2,) = ax.plot([], [], lw=2, label="Room 2")
(line_room3,) = ax.plot([], [], lw=2, label="Room 3")
(line_cooler,) = ax.plot([], [], lw=2, label="Controller")
annotation = plt.text(x=2400, y=1, s=f"Iter: {0}")
annotation.set_animated(True)
# annotation.set_bbox(dict(facecolor='0.8', alpha=0.5))


def init():
    line_room.set_data([], [])
    line_room2.set_data([], [])
    line_room3.set_data([], [])
    line_cooler.set_data([], [])

    # grids
    cax = ax
    cax.xaxis.set_minor_locator(AutoMinorLocator())
    cax.yaxis.set_minor_locator(AutoMinorLocator())
    cax.grid(
        which="major",
        axis="both",
        linestyle="--",
        linewidth=0.5,
        color="black",
        zorder=0,
    )
    cax.grid(
        which="minor", axis="both", linestyle="--", linewidth=0.5, color="0.7", zorder=0
    )

    # auxiliary
    cax.set_ylim([11.85, 21.85])
    cax.set_xlim([0, 13])
    cax.legend()
    # cax.get_legend().remove()
    cax.set_ylabel("Temperature / $°C$")

    # ticks
    # xticks = np.arange(mpc_log.index[0], mpc_log.index[-1] + 1, 24 * 3600)
    # xtickval = [str(i) for i, _ in enumerate(xticks)]
    # plt.xticks(xticks, xtickval)
    cax.set_xlabel("Time / h")

    # annotation

    return line_room, line_room2, line_room3, line_cooler, annotation


########################### CREATE PLOTS #####################################


def animate(i=0):  # Upper plot: Temperatures
    # plots
    # mpc_log.plot(y='T_0', ax=ax, label='temperature')
    room_mDot1 = admm_at_time_step(data=room_results1, time_step=time_step, iteration=i)
    x_time = room_mDot1.index
    x_time = x_time[3:,]
    room_mDot_var1 = room_mDot1["variable"]["T_v"]
    room_mDot_var1 = room_mDot_var1.iloc[3:,]
    line_room.set_data(x_time.values / 3600, room_mDot_var1.values - 273.15)
    room_mDot2 = admm_at_time_step(data=room_results2, time_step=time_step, iteration=i)
    room_mDot_var2 = room_mDot2["variable"]["T_v"]
    room_mDot_var2 = room_mDot_var2.iloc[3:,]
    line_room2.set_data(x_time.values / 3600, room_mDot_var2.values - 273.15)
    room_mDot3 = admm_at_time_step(data=room_results2, time_step=time_step, iteration=i)
    room_mDot_var3 = room_mDot3["variable"][["T_v"]]
    room_mDot_var3 = room_mDot_var3.iloc[3:,]
    line_room3.set_data(x_time.values / 3600, room_mDot_var3.values - 273.15)
    cooler_mDot = admm_at_time_step(
        data=cooler_results, time_step=time_step, iteration=i
    )
    cooler_mDot_var = cooler_mDot["variable"]["T_v_out"]
    cooler_mDot_var = cooler_mDot_var.iloc[1:,]
    line_cooler.set_data(x_time.values / 3600, cooler_mDot_var.values - 273.15)

    # annotation
    annotation.set_text(s=f"Iter: {i}")

    return line_room, line_room2, line_room3, line_cooler, annotation


def main():
    setup_figure()
    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=40,
        interval=800,
        blit=True,
        repeat_delay=1500,
    )
    anim.save("admm_animation3.gif", writer="imagemagick")
    if filename:
        plt.savefig(fname=filename, format=file_format)


def png_main():
    setup_figure()
    init()
    animate(10)
    if filename:
        plt.savefig(fname=filename, format=file_format)


if __name__ == "__main__":
    main()
