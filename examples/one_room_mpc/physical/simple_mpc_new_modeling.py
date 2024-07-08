import logging
import os
from pathlib import Path
from typing import List
import json
import requests

from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from agentlib.utils.multi_agent_system import LocalMASAgency

from agentlib_mpc.models.casadi_model_new import CasadiModel2, CasadiModelConfig2, Var
from agentlib_mpc.utils.analysis import load_mpc_stats
from agentlib_mpc.utils.plotting.interactive import show_dashboard

logger = logging.getLogger(__name__)

# script variables
ub = 295.15


class MyCasadiModelConfig(CasadiModelConfig2):

    # controls
    mDot = Var.input(
        value=0.0225,
        unit="m³/s",
        description="Air mass flow into zone",
    )
    # disturbances
    load = Var.input(
        value=150,
        unit="W",
        description="Heat load into zone"
    )
    T_in = Var.input(
        value=290.15,
        unit="K",
        description="Inflow air temperature"
    )
    # settings
    T_upper = Var.input(
        name="T_upper",
        value=294.15,
        unit="K",
        description="Upper boundary (soft) for T.",
    )

    # states
    T = Var.state(
        value=293.15,
        unit="K",
        description="Temperature of zone"
    )
    T_slack = Var.state(
        value=0,
        unit="K",
        description="Slack variable of temperature of zone"
    )

    # parameters
    cp = Var.parameter(
        value=1000,
        unit="J/kg*K",
        description="thermal capacity of the air"
    )
    C = Var.parameter(
        value=100000,
        unit="J/K",
        description="thermal capacity of zone"
    )
    s_T = Var.parameter(
        value=1,
        unit="-",
        description="Weight for T in constraint function"
    )
    r_mDot = Var.parameter(
        value=1,
        unit="-",
        description="Weight for mDot in objective function"
    )

    # outputs
    T_out = Var.output(
        unit="K",
        description="Temperature of zone"
    )

    def setup_system(self):
        # Define ode
        self.T.ode = (
            self.cp * self.mDot / self.C * (self.T_in - self.T) + self.load / self.C
        )

        # Define ae
        self.T_out.alg = self.T  # math operation to get the symbolic variable

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # soft constraints
            (0, self.T + self.T_slack, self.T_upper),
        ]

        # Objective function
        self.cost_function = sum(
            [
                self.r_mDot * self.mDot,
                self.s_T * self.T_slack**2,
            ]
        )

class MyCasadiModel(CasadiModel2):
    config: MyCasadiModelConfig




ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 60}

AGENT_MPC = {
    "id": "myMPCAgent",
    "modules": [
        {"module_id": "Ag1Com", "type": "local_broadcast"},
        {
            "module_id": "myMPC",
            "type": "agentlib_mpc.mpc",
            "optimization_backend": {
                "type": "casadi",
                "model": {"type": {"file": __file__, "class_name": "MyCasadiModel"}},
                "discretization_options": {
                    "collocation_order": 2,
                    "collocation_method": "legendre",
                },
                "solver": {
                    "name": "ipopt",
                    "options": {"ipopt.print_level": 5}
                },
                "results_file": "results//mpc.csv",
                "save_results": True,
                "overwrite_result_file": True,
            },
            "time_step": 300,
            "prediction_horizon": 15,
            "parameters": [
                {"name": "s_T", "value": 3},
                {"name": "r_mDot", "value": 1},
            ],
            "inputs": [
                {"name": "load", "value": 150},
                {"name": "T_upper", "value": ub},
                {"name": "T_in", "value": 290.15},
            ],
            "controls": [{"name": "mDot", "value": 0.02, "ub": 0.05, "lb": 0}],
            "outputs": [{"name": "T_out"}],
            "states": [
                {
                    "name": "T",
                    "value": 298.16,
                    "ub": 303.15,
                    "lb": 288.15,
                    "alias": "T",
                    "source": "SimAgent",
                }
            ],
        },
    ],
}
AGENT_SIM = {
    "id": "SimAgent",
    "modules": [
        {"module_id": "Ag1Com", "type": "local_broadcast"},
        {
            "module_id": "room",
            "type": "simulator",
            "model": {
                "type": {"file": __file__, "class_name": "MyCasadiModel"},
                "states": [{"name": "T", "value": 298.16}],
            },
            "t_sample": 10,
            "update_inputs_on_callback": False,
            "save_results": True,
            "outputs": [
                {"name": "T_out", "value": 298, "alias": "T"},
            ],
            "inputs": [
                {"name": "mDot", "value": 0.02, "alias": "mDot"},
            ],
        },
    ],
}


def run_example(
    with_plots=True, log_level=logging.INFO, until=10000, with_dashboard=False
):
    # Change the working directly so that relative paths work
    os.chdir(Path(__file__).parent)

    # Set the log-level
    logging.basicConfig(level=log_level)
    mas = LocalMASAgency(
        agent_configs=[AGENT_MPC, AGENT_SIM], env=ENV_CONFIG, variable_logging=False
    )
    mas.run(until=until)
    try:
        stats = load_mpc_stats("results/stats_mpc.csv")
    except FileNotFoundError:
        stats = None
    results = mas.get_results(cleanup=True)
    mpc_results = results["myMPCAgent"]["myMPC"]
    sim_res = results["SimAgent"]["room"]

    if with_dashboard:

        show_dashboard(mpc_results, stats)

    if with_plots:
        import matplotlib.pyplot as plt
        from agentlib_mpc.utils.plotting.mpc import plot_mpc

        fig, ax = plt.subplots(2, 1, sharex=True)
        t_sim = sim_res["T_out"]
        t_sample = t_sim.index[1] - t_sim.index[0]
        aie_kh = (t_sim - ub).abs().sum() * t_sample / 3600
        energy_cost_kWh = (
            (sim_res["mDot"] * (sim_res["T_out"] - sim_res["T_in"])).sum()
            * t_sample
            * 1
            / 3600
        )  # cp is 1
        print(f"Absoulute integral error: {aie_kh} Kh.")
        print(f"Cooling energy used: {energy_cost_kWh} kWh.")

        plot_mpc(
            series=mpc_results["variable"]["T"] - 273.15,
            ax=ax[0],
            plot_actual_values=True,
            plot_predictions=True,
        )
        ax[0].axhline(ub - 273.15, color="grey", linestyle="--", label="upper boundary")
        plot_mpc(
            series=mpc_results["variable"]["mDot"],
            ax=ax[1],
            plot_actual_values=True,
            plot_predictions=True,
        )

        ax[1].legend()
        ax[0].legend()
        ax[0].set_ylabel("$T_{room}$ / °C")
        ax[1].set_ylabel("$\dot{m}_{air}$ / kg/s")
        ax[1].set_xlabel("simulation time / s")
        ax[1].set_ylim([0, 0.06])
        ax[1].set_xlim([0, until])
        plt.show()

    return results


def run_example_clonemap():
    # set up full example using regular clonemap
    URL = "http://localhost:30009/api/clonemap/mas"
    CFG_PATH = Path(__file__).parent.joinpath("simple_mpc_clonemap_config.json")
    with open(CFG_PATH, "r") as file:
        DATA = json.load(file)
    requests.post(URL, json=DATA)

    return 0


if __name__ == "__main__":
    run_example(
        with_plots=False, with_dashboard=True, until=7200, log_level=logging.WARNING
    )
    # run_example_clonemap()