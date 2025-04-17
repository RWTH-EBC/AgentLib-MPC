"""
Example for running a multi-agent-system performing a distributed MPC with
ALADIN. Creates three agents, one for the AHU, one for a supplied room and one
for simulating the system.
"""

import os
import logging
from typing import List

from agentlib.utils.multi_agent_system import LocalMASAgency


from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from agentlib_mpc.utils.plotting.mpc_dashboard import launch_dashboard_from_results


class CaCoolerConfig(CasadiModelConfig):
    inputs: list[CasadiInput] = [
        # controls
        CasadiInput(
            name="mDot",
            value=0.0225,
            unit="kg/s",
            description="Air " "mass flow out of cooler.",
        ),
    ]

    states: list[CasadiState] = [
        # differential
        # algebraic
        # slack variables
    ]

    parameters: list[CasadiParameter] = [
        CasadiParameter(
            name="r_mDot",
            value=1,
            unit="-",
            description="Weight for mDot in objective function",
        )
    ]

    outputs: list[CasadiOutput] = [
        CasadiOutput(
            name="mDot_out",
            value=0.0225,
            unit="kg/s",
            description="Air mass flow out of cooler.",
        ),
    ]


class CaCooler(CasadiModel):

    config: CaCoolerConfig

    def setup_system(self):
        # Define ode

        # Define ae
        self.mDot_out.alg = 1 * self.mDot

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # soft constraints
            # outputs
        ]

        # Objective function
        objective = sum(
            [
                self.r_mDot * self.mDot,
            ]
        )

        return objective


class CaCooledRoomConfig(CasadiModelConfig):
    inputs: list[CasadiInput] = [
        # couplings
        CasadiInput(
            name="mDot_0",
            value=0.0225,
            unit="kg/s",
            description="Air mass flow into zone 0",
        ),
        # disturbances
        CasadiInput(
            name="d_0", value=150, unit="W", description="Heat load into zone 0"
        ),
        CasadiInput(
            name="T_in",
            value=17.0,
            unit="°C",
            description="Inflow air temperature",  # Changed from 290.15K
        ),
        # settings
        CasadiInput(
            name="T_0_set",
            value=21.0,
            unit="°C",  # Changed from 294.15K
            description="Set point for T_0 in objective function",
        ),
        CasadiInput(
            name="T_0_upper",
            value=21.0,
            unit="°C",  # Changed from 294.15K
            description="Upper boundary (soft) for T_0.",
        ),
    ]

    states: list[CasadiState] = [
        # differential
        CasadiState(
            name=f"T_0",
            value=20.0,
            unit="°C",
            description="Temperature of zone 0",  # Changed from 293.15K
        ),
        # algebraic
        # slack variables
        CasadiState(
            name=f"T_0_slack",
            value=0,
            unit="°C",
            description="Slack variable of temperature of zone 0",
        ),
    ]

    parameters: list[CasadiParameter] = [
        CasadiParameter(
            name="cp",
            value=1000,
            unit="J/kg*K",
            description="thermal capacity of the air",
        ),
        CasadiParameter(
            name="c_0",
            value=100000,
            unit="J/kg*K",
            description="thermal capacity of zone 0",
        ),
        CasadiParameter(
            name="q_T_0",
            value=1,
            unit="-",
            description="Weight for T_0 in objective function",
        ),
        CasadiParameter(
            name="s_T_0",
            value=1,
            unit="-",
            description="Weight for T_0 in constraint function",
        ),
    ]


class CaCooledRoom(CasadiModel):

    config: CaCooledRoomConfig

    def setup_system(self):
        # Define ode
        self.T_0.ode = (
            self.cp * self.mDot_0 / self.c_0 * (self.T_in - self.T_0)
            + self.d_0 / self.c_0
        )

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # soft constraints
            (0, self.T_0 + self.T_0_slack, self.T_0_upper),
            # outputs
        ]

        # Objective function
        objective = sum(
            [
                self.s_T_0 * self.T_0_slack**2,
                self.q_T_0 * (self.T_0 - self.T_0_set) ** 2,
            ]
        )

        return objective


class FullModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        # controls
        CasadiInput(
            name="mDot_0",
            value=0.0225,
            unit="kg/s",
            description="Air mass flow into zone 0",
        ),
        # disturbances
        CasadiInput(
            name="d_0", value=150, unit="W", description="Heat load into zone 0"
        ),
        CasadiInput(
            name="T_in",
            value=17.0,
            unit="°C",
            description="Inflow air temperature",  # Changed from 290.15K
        ),
        # settings
        CasadiInput(
            name="T_0_set",
            value=21.0,
            unit="°C",  # Changed from 294.15K
            description="Set point for T_0 in objective function",
        ),
        CasadiInput(
            name="T_0_upper",
            value=21.0,
            unit="°C",  # Changed from 294.15K
            description="Upper boundary (soft) for T_0.",
        ),
    ]

    states: List[CasadiState] = [
        # differential
        CasadiState(
            name=f"T_0",
            value=20.0,
            unit="°C",
            description="Temperature of zone 0",  # Changed from 293.15K
        ),
        # algebraic
        # slack variables
        CasadiState(
            name=f"T_0_slack",
            value=0,
            unit="°C",
            description="Slack variable of temperature of zone 0",
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
            name="c_0",
            value=100000,
            unit="J/kg*K",
            description="thermal capacity of zone 0",
        ),
        CasadiParameter(
            name="q_T_0",
            value=1,
            unit="-",
            description="Weight for T_0 in objective function",
        ),
        CasadiParameter(
            name="s_T_0",
            value=1,
            unit="-",
            description="Weight for T_0 in constraint function",
        ),
        CasadiParameter(
            name="r_mDot_0",
            value=1,
            unit="-",
            description="Weight for mDot_0 in objective function",
        ),
    ]
    outputs: List[CasadiOutput] = [
        CasadiOutput(name="T_0_out", unit="°C", description="Temperature of zone 0")
    ]


class FullModel(CasadiModel):

    config: FullModelConfig

    def setup_system(self):
        # Define ode
        self.T_0.ode = (
            self.cp * self.mDot_0 / self.c_0 * (self.T_in - self.T_0)
            + self.d_0 / self.c_0
        )

        # Define ae
        self.T_0_out.alg = self.T_0 * 1


def make_configs() -> List[dict]:
    simulator = {
        "id": "Simulation",
        "modules": [
            {
                "module_id": "simulator",
                "type": "simulator",
                "model": {
                    "type": {
                        "file": __file__,
                        "class_name": "FullModel",
                    },
                    "states": [{"name": "T_0", "value": 25.01}],  # Changed from 298.16K
                },
                "t_sample": 15,
                "save_results": True,
                "outputs": [
                    {"name": "T_0_out", "value": 25.01, "alias": "T_0"}
                ],  # Changed from 298.16K
                "inputs": [{"name": "mDot_0", "value": 0.02, "alias": "mDot"}],
            },
            {"type": "local_broadcast"},
        ],
    }
    coordinator = {
        "id": "Coordinator",
        "modules": [
            {
                "module_id": "coordinator",
                "type": "agentlib_mpc.aladin_coordinator",
                "prediction_horizon": 15,
                "time_step": 300,
                "sampling_time": 60,
                "penalty_factor": 100,
                "penalty_variation_factor": 1.2,
                "wait_time_on_start_iters": 0.2,
                "registration_period": 5,
                "iter_max": 15,
                "qp_penalty": 10,
                "qp_penalty_variation_factor": 1.2,
                "qp_step_size": 0.8,
                "qp_solver": "proxqp",
                "save_solve_stats": True,
                "solve_stats_file": "results//residuals.csv",
            },
            {"module_id": "AgCom", "type": "local_broadcast"},
        ],
    }

    cooler = {
        "id": "Cooler",
        "modules": [
            {
                "module_id": "mpc",
                "type": "agentlib_mpc.aladin",
                "optimization_backend": {
                    "type": "casadi_aladin",
                    "model": {
                        "type": {
                            "file": __file__,
                            "class_name": "CaCooler",
                        }
                    },
                    # "discretization_options": {
                    #     "collocation_order": 2,
                    #     "collocation_method": "legendre",
                    # },
                    "discretization_options": {
                        "method": "multiple_shooting",
                        "integrator": "euler",
                    },
                    "solver": {"name": "ipopt", "options": {"ipopt.print_level": 0}},
                    "results_file": "results//cooler_res.csv",
                    "overwrite_result_file": True,
                },
                "coordinator": {"agent_id": "Coordinator"},
                "time_step": 240,
                "prediction_horizon": 3,
                "max_iterations": 20,
                "parameters": [{"name": "r_mDot", "value": 1}],
                "inputs": [],
                "controls": [{"name": "mDot", "value": 0.02, "ub": 0.1, "lb": 0}],
                "states": [],
                "couplings": [
                    {"name": "mDot_out", "alias": "mDotCoolAir", "value": 0.05}
                ],
                "r_del_u": {"mDot": 10},
            },
            {"type": "local_broadcast"},
        ],
    }

    room = {
        "id": "CooledRoom",
        "modules": [
            {"type": "local_broadcast"},
            {
                "module_id": "mpc",
                "type": "agentlib_mpc.aladin",
                "optimization_backend": {
                    "type": "casadi_aladin",
                    "model": {
                        "type": {
                            "file": __file__,
                            "class_name": "CaCooledRoom",
                        }
                    },
                    # "discretization_options": {
                    #     "collocation_order": 2,
                    #     "collocation_method": "legendre",
                    # },
                    "discretization_options": {
                        "method": "multiple_shooting",
                        "integrator": "euler",
                    },
                    "solver": {"name": "ipopt", "options": {"ipopt.print_level": 0}},
                    "results_file": "results//room.csv",
                    "overwrite_result_file": True,
                },
                "coordinator": {"agent_id": "Coordinator"},
                "time_step": 240,
                "prediction_horizon": 3,
                "max_iterations": 20,
                "parameters": [
                    # {"name": "q_T_0", "value": 0},
                    {"name": "s_T_0", "value": 1},
                ],
                "inputs": [
                    {"name": "d_0", "value": 150},
                    {"name": "T_0_set", "value": 21.0},  # Changed from 294.55K
                    {"name": "T_0_upper", "value": 21.0},  # Changed from 294.15K
                    {"name": "T_in", "value": 17.0},  # Changed from 290.15K
                ],
                "controls": [],
                "states": [
                    {
                        "name": "T_0",
                        "value": 25.01,
                        "ub": 30.0,
                        "lb": 15.0,
                    }  # Changed from Kelvin values
                ],
                "couplings": [
                    {
                        "name": "mDot_0",
                        "alias": "mDotCoolAir",
                        "value": 0.05,
                        "ub": 0.1,
                        "lb": 0,
                    }
                ],
            },
        ],
    }
    return [cooler, room, coordinator, simulator]


def plot(results, start_pred=0):
    import matplotlib.pyplot as plt
    from agentlib_mpc.utils.analysis import admm_at_time_step

    res_sim = results["Simulation"]["simulator"]
    mpc_room_results = results["CooledRoom"]["mpc"]

    room_res = admm_at_time_step(
        data=mpc_room_results, time_step=start_pred, iteration=-1
    )

    fig, ax = plt.subplots(2, 1)
    ax[0].axhline(21.0, label="reference value", ls="--")  # Changed from 294.55
    ax[0].plot(res_sim["T_0_out"], label="temperature")
    ax[0].plot(room_res["variable"]["T_0"], label="temperature prediction")
    ax[1].plot(res_sim["mDot_0"], label="air mass flow")
    ax[1].legend()
    ax[0].legend()
    ax[0].set_ylabel("Temperature (°C)")
    ax[1].set_ylabel("Mass flow (kg/s)")
    ax[0].set_title("Room Temperature Control with ALADIN")
    plt.show()


def run_example(
    until=3000, with_plots=True, start_pred=0, log_level=logging.INFO, cleanup=True
):
    # Set the log-level
    logging.basicConfig(level=log_level)

    # Change the working directly so that relative paths work
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    env_config = {"rt": False, "factor": 0.08, "t_sample": 600}
    mas = LocalMASAgency(
        agent_configs=make_configs(), env=env_config, variable_logging=False
    )
    mas.run(until=until)
    results = mas.get_results(cleanup=cleanup)

    if with_plots:
        plot(results, start_pred=0)
        launch_dashboard_from_results(results)
    return results


if __name__ == "__main__":
    run_example(
        with_plots=True,
        until=3600,
        start_pred=0,
        cleanup=False,
        log_level=logging.INFO,
    )
