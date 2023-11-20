import json
import os
import logging
from agentlib.utils.multi_agent_system import LocalMASAgency

agent_configs = [
    "configs//rlt_admm.json",
    "configs//room_1_admm.json",
    "configs//room_2_admm.json",
    "configs//room_3_admm.json",
    "configs//room_4_admm.json",
    "configs//simulation//simulator_agent.json",
    "configs//coordinator.json",
]


def run_example(
    until=3000,
    with_plots=True,
    log_level=logging.INFO,
    cleanup: bool = True,
):
    # Set the log-level
    logging.basicConfig(level=log_level)

    # Change the working directly so that relative paths work
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    env_config = {"rt": False, "t_sample": 60}
    conf_dicts = []

    for conf in agent_configs:
        with open(conf) as f:
            conf_dict = json.load(f)
        modules = conf_dict["modules"]
        for i, mod in enumerate(modules):
            if isinstance(mod, str):
                with open(mod) as f:
                    mod = json.load(f)
            if mod["type"] == "agentlib_mpc.admm_local":
                mod["type"] = "agentlib_mpc.admm_coordinated"
        conf_dicts.append(conf_dict)

    mas = LocalMASAgency(
        agent_configs=conf_dicts, env=env_config, variable_logging=False
    )
    mas.run(until=until)
    results = mas.get_results(cleanup=cleanup)

    if with_plots:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1)
        ax[1].set_ylim([0, 0.05])
        ax[0].axhline(296, label="reference value")

        for i in range(1, 5):
            sim_res = results["Simulation"][f"room_{i}"]
            ax[0].plot(sim_res["T_out"], label=f"temperature_{i}")
            ax[1].plot(sim_res["mDot"], label=f"mDot_{i}")

        # ax[1].plot(res_sim["mDot"], label="air mass flow")
        # ax[1].legend()
        # ax[0].legend()
        plt.show()

    return results


if __name__ == "__main__":
    run_example(
        with_plots=True,
        until=1800,
        cleanup=False,
        log_level=logging.INFO,
    )
