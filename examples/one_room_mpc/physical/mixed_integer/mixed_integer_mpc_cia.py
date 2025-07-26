import logging
import os
from pathlib import Path
from typing import List

import casadi as ca

from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from agentlib.utils.multi_agent_system import LocalMASAgency

logger = logging.getLogger(__name__)


# script variables
ub = 295.15

# constants
COOLING = 1000


class MyCasadiModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        # controls
        CasadiInput(
            name="cooling_power",
            value=400,
            unit="W",
            description="Air mass flow " "into zone",
        ),
        CasadiInput(
            name="cooler_on",
            value=1,
            unit="-",
            description="On / off signal of mass flow.",
            lb=0,
            ub=1,
        ),
        # disturbances
        CasadiInput(
            name="load", value=150, unit="W", description="Heat " "load into zone"
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
            name="r_cooling",
            value=1 / 5,
            unit="-",
            description="Weight for mDot in objective function",
        ),
        CasadiParameter(
            name="cooler_mod_limit",
            value=200,
            unit="W",
            description="Cooling power cannot modulate below this value",
        ),
    ]
    outputs: List[CasadiOutput] = [
        CasadiOutput(name="T_out", unit="K", description="Temperature of zone")
    ]


class MyCasadiModel(CasadiModel):
    config: MyCasadiModelConfig

    def setup_system(self):
        # Define ode
        self.T.ode = (self.load - self.cooling_power) / self.C

        # Define ae
        self.T_out.alg = self.T  # math operation to get the symbolic variable

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # bigM reformulation
            (-ca.inf, self.cooling_power - self.cooler_on * COOLING, 0),
            (0, self.cooling_power - self.cooler_on * self.cooler_mod_limit, ca.inf),
            # soft constraints
            (0, self.T + self.T_slack, self.T_upper),
        ]

        # Objective function
        objective = sum(
            [
                self.r_cooling * self.cooling_power,
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
            "type": "agentlib_mpc.minlp_mpc",
            "optimization_backend": {
                "type": "casadi_cia",
                "model": {"type": {"file": __file__, "class_name": "MyCasadiModel"}},
                "discretization_options": {
                    "collocation_order": 2,
                    "collocation_method": "legendre",
                },
                "solver": {
                    "name": "ipopt",
                },
                "overwrite_result_file": True,
                "results_file": "results//mpc.csv",
                "save_results": True
            },
            "time_step": 300,
            "prediction_horizon": 10,
            "parameters": [
                {"name": "s_T", "value": 3},
                {"name": "r_cooling", "value": 1 / 3},
                {"name": "cooler_mod_limit", "value": 250},
            ],
            "inputs": [
                {"name": "load", "value": 150},
                {"name": "T_upper", "value": ub},
                {"name": "T_in", "value": 290.15},
            ],
            "controls": [
                {"name": "cooling_power", "value": 250, "ub": 500, "lb": 0},
            ],
            "binary_controls": [
                {"name": "cooler_on", "value": 0, "ub": 1, "lb": 0},
            ],
            "states": [{"name": "T", "value": 298.16, "ub": 303.15, "lb": 288.15}],
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
            "outputs": [
                {"name": "T_out", "value": 298, "alias": "T"},
            ],
            "inputs": [
                {"name": "cooling_power", "value": 200, "alias": "cooling_power"},
            ],
        },
    ],
}


def run_example(with_plots=True, log_level=logging.INFO, until=10000):
    # Change the working directly so that relative paths work
    os.chdir(Path(__file__).parent)

    # Set the log-level
    logging.basicConfig(level=log_level)
    mas = LocalMASAgency(
        agent_configs=[AGENT_MPC, AGENT_SIM], env=ENV_CONFIG, variable_logging=True
    )
    mas.run(until=until)
    results = mas.get_results()
    if with_plots:
        # show_dashboard(results["myMPCAgent"]["myMPC"])

        import matplotlib.pyplot as plt
        from agentlib_mpc.utils.plotting.mpc import plot_mpc

        fig, ax = plt.subplots(3, 1, sharex=True)
        mpc_results = results["myMPCAgent"]["myMPC"]

        fig.suptitle("CIA")
        plot_mpc(
            series=mpc_results["variable"]["T"] - 273.15,
            ax=ax[0],
            plot_actual_values=True,
            plot_predictions=True,
        )
        ax[0].axhline(ub - 273.15, color="grey", linestyle="--", label="upper boundary")
        plot_mpc(
            series=mpc_results["variable"]["cooling_power"],
            ax=ax[1],
            plot_actual_values=True,
            plot_predictions=True,
            step=True,
        )
        ax[1].axhline(250, color="grey", linestyle="--", label="modulation limit")
        # ax[1].plot(
        #     results["myMPCAgent"]["AgentLogger"]["cooling_power"], label="air mass flow"
        # )
        plot_mpc(
            series=mpc_results["variable"]["cooler_on"],
            ax=ax[2],
            plot_actual_values=True,
            plot_predictions=True,
            step=True,
        )
        ax[0].set_ylabel("$T_{room}$ / °C")
        ax[1].set_ylabel("$\dot{Q}_{cool}$ / W")
        ax[2].set_ylabel("Switch")
        ax[2].set_xlabel("simulation time / s")
        ax[2].set_yticks([0, 1])
        ax[1].set_ylim([0, 510])
        # results["myMPCAgent"]["AgentLogger"]["cooler_on"].plot(ax=ax[2], color="black", drawstyle="steps-post", label="Switch")
        # ax[2].set_xlim([0, until])
        ax[1].legend()
        ax[0].legend()
        ax[2].legend()
        plt.show()

    return results


if __name__ == "__main__":
    run_example(with_plots=True, until=3600)
