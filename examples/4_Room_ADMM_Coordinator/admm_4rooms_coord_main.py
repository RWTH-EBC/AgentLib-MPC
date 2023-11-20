import os
import logging
from agentlib.utils.multi_agent_system import LocalMASAgency

agent_configs = [
    "configs\\rlt_admm.json",
    "configs\\room_1_admm.json",
    "configs\\room_2_admm.json",
    "configs\\room_3_admm.json",
    "configs\\room_4_admm.json",
    "configs\\coordinator.json",
    "configs\\simulation\\simulator_agent.json",
]


def run_example(
    until=3000, with_plots=True, log_level: int = logging.INFO, cleanup: bool = True
):
    # Set the log-level
    logging.basicConfig(level=log_level)

    # Change the working directory so that relative paths work
    os.chdir(os.path.dirname(__file__))

    env_config = {"rt": False, "t_sample": 60}
    mas = LocalMASAgency(
        agent_configs=agent_configs, env=env_config, variable_logging=False
    )
    mas.run(until=until)
    results = mas.get_results(cleanup=cleanup)

    if with_plots:
        import matplotlib.pyplot as plt
        from agentlib_mpc.utils.analysis import admm_at_time_step

        nRooms = 4
        plotPredTimeStep = [120, 1500]

        # remove too large values
        for value in plotPredTimeStep:
            if value > until:
                plotPredTimeStep.remove(value)

        def mDotSumSim():
            mDotSum = 0
            for i in range(1, nRooms + 1):
                mDotSum = mDotSum + results["Simulation"][f"room_{i}"]["mDot"]
            return mDotSum

        def mDotSumPred(j):
            mDotSum = 0
            for i in range(1, nRooms + 1):
                room_res = admm_at_time_step(
                    data=results[f"room_{i}"]["admm_module"], time_step=j, iteration=-1
                )
                mDotSum = mDotSum + room_res["variable"]["mDot"]
            return mDotSum

        colorRoom = ["b", "g", "r", "c", "m", "k", "y"]
        fig, ax = plt.subplots(2, 1)
        # ax[1].set_ylim([-0.002, 0.052])
        ax[0].axhline(296, label="reference value")
        ax[1].set_xlabel("time in sec")
        ax[0].set_ylabel("room temperature in K")
        ax[1].set_ylabel("mDot")

        for i in range(1, nRooms + 1):  # [1]: #
            sim_res = results["Simulation"][f"room_{i}"]
            ax[0].plot(
                sim_res["T_out"],
                color=colorRoom[i - 1],
                marker=".",
                label=f"temperature_{i}",
            )
            ax[1].plot(
                sim_res["mDot"], color=colorRoom[i - 1], marker=".", label=f"mDot_{i}"
            )

            for j in plotPredTimeStep:  #
                room_res = admm_at_time_step(
                    data=results[f"room_{i}"]["admm_module"], time_step=j, iteration=-1
                )
                room_res.index = room_res.index + env_config["t_sample"]
                ax[0].plot(
                    room_res["variable"]["T"],
                    color=colorRoom[i - 1],
                    label=f"temperature {i} at {j} pred",
                )
                ax[1].plot(
                    room_res["variable"]["mDot"],
                    color=colorRoom[i - 1],
                    label=f"air mass flow {i} at {j} pred",
                )

        ax[1].plot(mDotSumSim(), marker="x", label="sum mDot")
        [
            ax[1].plot(
                mDotSumPred(j), marker="x", label=f"total air mass flow at {j} pred"
            )
            for j in plotPredTimeStep
        ]
        ax[0].legend()
        ax[1].legend()
        plt.show()

    return results


if __name__ == "__main__":
    run_example(with_plots=True, until=1800, cleanup=False, log_level=logging.WARNING)
