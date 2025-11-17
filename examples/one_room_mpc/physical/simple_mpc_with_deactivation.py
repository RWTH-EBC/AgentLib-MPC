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
            name="mpc_active",
            value=1,
            unit="1",
            description="Flag, whether mpc or default control is used.",
        ),
        CasadiParameter(
            name="mDot_default",
            value=0.025,
            unit="kg/s",
            description="Default mass flow.",
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
        mDot = self.mDot * self.mpc_active + self.mDot_default * (1 - self.mpc_active)

        # Define ode
        self.T.ode = self.cp * mDot / self.C * (self.T_in - self.T) + self.load / self.C

        # Define ae
        self.T_out.alg = self.T

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # soft constraints
            (0, self.T + self.T_slack, self.T_upper),
        ]

        # Objective function
        objective = sum(
            [
                self.r_mDot * mDot,
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
            "module_id": "skip_mpc",
            "type": "agentlib_mpc.skip_mpc_intervals",
            "intervals": [[30, 35], [50, 55]],
            "time_unit": "minutes",
            "log_level": "debug",
            "public_active_message": {
                "name": "public_active_message",
                "alias": "mpc_active",
                "value": True,
            },
            "public_inactive_message": {
                "name": "public_inactive_message",
                "alias": "mpc_active",
                "value": False,
            },
        },
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
            "enable_deactivation": True,
            "deactivation_source": {"module_id": "skip_mpc"},
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
            "result_causalities": ["input", "parameter", "local", "output"],
            "outputs": [
                {"name": "T_out", "value": 298, "alias": "T"},
            ],
            "inputs": [
                {"name": "mDot", "value": 0.02, "alias": "mDot"},
            ],
            "parameters": [{"name": "mpc_active", "value": True}],
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
        agent_configs=[AGENT_MPC, AGENT_SIM], env=ENV_CONFIG, variable_logging=True
    )
    mas.run(until=until)
    try:
        stats = load_mpc_stats("results/__mpc.csv")
    except Exception:
        stats = None
    results = mas.get_results(cleanup=False)
    mpc_results = results["myMPCAgent"]["myMPC"]
    sim_res = results["SimAgent"]["room"]

    if with_plots:
        # Pass the full results to plot
        plot(sim_res, until, results)

    if with_dashboard:
        show_dashboard(mpc_results, stats)

    return results


def plot(sim_res: pd.DataFrame, until: float, results=None):
    import matplotlib.pyplot as plt

    # Calculate performance metrics
    t_sim = sim_res["T_out"]
    t_sample = t_sim.index[1] - t_sim.index[0]
    aie_kh = (t_sim - ub).abs().sum() * t_sample / 3600
    energy_cost_kWh = (
        (sim_res["mDot"] * (sim_res["T_out"] - sim_res["T_in"])).sum()
        * t_sample
        * 1
        / 3600
    )  # cp is 1
    print(f"Absolute integral error: {aie_kh} Kh.")
    print(f"Cooling energy used: {energy_cost_kWh} kWh.")

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 10))

    # Plot room temperature (from simulator results)
    ax[0].plot(
        sim_res.index,
        sim_res["T_out"] - 273.15,
        "b-",
        linewidth=2,
        label="Room Temperature",
    )
    ax[0].axhline(
        ub - 273.15, color="red", linestyle="--", linewidth=1.5, label="Upper Boundary"
    )
    ax[0].set_ylabel("Temperature (°C)", fontsize=12)
    ax[0].legend(fontsize=10)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title("Room Temperature Control System Monitoring", fontsize=14)

    # Plot mass flow (from simulator results)
    ax[1].plot(sim_res.index, sim_res["mDot"], "g-", linewidth=2, label="Air Mass Flow")
    ax[1].set_ylabel("Mass Flow (kg/s)", fontsize=12)
    ax[1].set_ylim([0, 0.06])
    ax[1].legend(fontsize=10)
    ax[1].grid(True, alpha=0.3)

    agent_logger = results["myMPCAgent"]["AgentLogger"]
    mpc_flag = agent_logger["MPC_FLAG_ACTIVE"]
    ax[2].step(
        agent_logger.index,
        mpc_flag,
        "r-",
        linewidth=2,
        where="post",
        label="MPC Active",
    )

    ax[2].set_ylim([-0.1, 1.1])
    ax[2].set_yticks([0, 1])
    ax[2].set_yticklabels(["Inactive", "Active"])
    ax[2].set_ylabel("MPC Status", fontsize=12)
    ax[2].set_xlabel("Simulation Time (s)", fontsize=12)
    ax[2].grid(True, alpha=0.3)

    # Set x limits for all subplots
    for a in ax:
        a.set_xlim([0, until])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_example(
        with_plots=True, with_dashboard=True, until=7200, log_level=logging.INFO
    )
