import logging
import os
from pathlib import Path
from typing import List

import pandas as pd

from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from agentlib.utils.multi_agent_system import LocalMASAgency

from agentlib_mpc.utils.analysis import load_mpc_stats
from agentlib_mpc.utils.plotting.interactive import show_dashboard

logger = logging.getLogger(__name__)

# script variables
ub = 295.15


class MyCasadiModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        # controls
        CasadiInput(
            name="mDot",
            value=0.0225,
            unit="m³/s",
            description="Air mass flow into zone",
        ),
        # disturbances
        CasadiInput(
            name="load", value=150, unit="W", description="Heat load into zone"
        ),
        CasadiInput(
            name="T_in", value=290.15, unit="K", description="Inflow air temperature"
        ),
        # settings
        CasadiInput(
            name="T_upper",
            value=294.15,
            unit="K",
            description="Upper boundary (soft) for T.",
        ),
    ]

    states: List[CasadiState] = [
        # differential
        CasadiState(
            name="T", value=293.15, unit="K", description="Temperature of zone"
        ),
        # algebraic
        # slack variables
        CasadiState(
            name="T_slack",
            value=0,
            unit="K",
            description="Slack variable of temperature of zone",
        ),
    ]

    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="cp",
            value=1000,
            unit="J/kg*K",
            description="thermal capacity of the air",
        ),
        CasadiParameter(
            name="C", value=100000, unit="J/K", description="thermal capacity of zone"
        ),
        CasadiParameter(
            name="s_T",
            value=1,
            unit="-",
            description="Weight for T in constraint function",
        ),
        CasadiParameter(
            name="r_mDot",
            value=1,
            unit="-",
            description="Weight for mDot in objective function",
        ),
    ]
    outputs: List[CasadiOutput] = [
        CasadiOutput(name="T_out", unit="K", description="Temperature of zone")
    ]


class MyCasadiModel(CasadiModel):
    config: MyCasadiModelConfig

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
        objective = sum(
            [
                self.r_mDot * self.mDot,
                self.s_T * self.T_slack**2,
            ]
        )

        return objective


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
                    "name": "fatrop",  # use fatrop with casadi 3.6.6 for speedup
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
                {"name": "T_in", "value": 290.15},
                {"name": "load", "value": 150},
                {"name": "T_upper", "value": ub},
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
        stats = load_mpc_stats("results/__mpc.csv")
    except Exception:
        stats = None
    results = mas.get_results(cleanup=False)
    mpc_results = results["myMPCAgent"]["myMPC"]
    sim_res = results["SimAgent"]["room"]

    if with_dashboard:
        show_dashboard(mpc_results, stats)

    if with_plots:
        plot(mpc_results, sim_res, until)

    return results


def plot(mpc_results: pd.DataFrame, sim_res: pd.DataFrame, until: float):
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


if __name__ == "__main__":
    run_example(
        with_plots=True, with_dashboard=True, until=7200, log_level=logging.INFO
    )
