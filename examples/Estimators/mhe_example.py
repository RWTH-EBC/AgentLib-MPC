import logging
from typing import List

from agentlib.utils.multi_agent_system import LocalMASAgency
from agentlib_mpc.utils.analysis import mpc_at_time_step, last_vals_at_trajectory_index
from matplotlib import pyplot as plt

from agentlib_mpc.models.casadi_model import (
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiModelConfig,
    CasadiOutput,
    CasadiModel,
)

T_UPPER_23 = 23
TRUE_CAP_FACTOR = 5.5


class Inputs:
    controls = [
        CasadiInput(
            name="mDot", value=0.22, unit="kg/s", description="Air mass flow into zone"
        ),  # mdot is 0.025 for 12, 0.45 for 24
    ]
    disturbances = [
        CasadiInput(
            name="load", value=150, unit="W", description="Heat load into zone"
        ),
        CasadiInput(
            name="T_in", value=290.15, unit="K", description="Inflow air temperature"
        ),  # should be around 17°C
        CasadiInput(
            name="T_ambient",
            value=28,
            unit="K",
            description="Ambient air temperature",
        ),
    ]
    settings = [
        CasadiInput(
            name="T_upper",
            value=22,
            unit="K",
            description="Upper boundary (soft) for T.",
        ),
    ]

    all = controls + disturbances + settings


class States:
    differential = [
        CasadiState(name="T", value=22, unit="K", description="Temperature of zone"),
        CasadiState(
            name="T_wall", value=23, unit="K", description="Temperature of wall"
        ),
    ]

    slack_variables = [
        CasadiState(
            name="T_slack",
            value=0,
            unit="K",
            description="Slack variable of temperature of zone",
        ),
    ]
    all = differential + slack_variables


class Parameters:
    constants = [
        CasadiParameter(
            name="cp",
            value=1005,
            unit="J/kg*K",
            description="thermal capacity of the air",
        ),
        CasadiParameter(
            name="rho", value=1.20, unit="kg/m^3", description="density of air"
        ),
    ]

    room = [
        CasadiParameter(
            name="full_capacity_from_volume_factor",
            value=5.5,  # 3_418_902
            unit="-",
            description="factor by which the thermal capacity of the air is amplified "
            "to consider other capacities like wall, furniture etc.",
        ),
        CasadiParameter(
            name="C_Wall",
            value=4_569_348,
            unit="J/K",
            description="thermal capacity of hull",
        ),
        CasadiParameter(
            name="RZone_Wall",
            value=0.0129,
            unit="K/W",
            description="heat transfer coefficient between zone and hull",
        ),
        CasadiParameter(
            name="R_hull_amb",
            value=0.1128,
            unit="K/W",
            description="heat transfer coefficient between outside and hull",
        ),
        CasadiParameter(name="V", value=59, unit="m^3", description="Volume of zone"),
    ]
    weights = [
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
    all = constants + room + weights


class RNGRoomConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = Inputs.all
    states: List[CasadiState] = States.all

    parameters: List[CasadiParameter] = Parameters.all
    outputs: List[CasadiOutput] = [
        CasadiOutput(name="T_out", unit="K", description="Temperature of zone"),
        CasadiOutput(name="cooling"),
        CasadiOutput(name="power_wall2zone"),
    ]


class RNGRoom(CasadiModel):
    config_type: RNGRoomConfig

    def setup_system(self):
        # Define ode

        # power wall to zone
        power_wall2zone = (self.T_wall - self.T) / self.RZone_Wall

        air_cooling = self.cp * self.mDot * (self.T_in - self.T)

        C_zone = self.rho * self.cp * self.V * self.full_capacity_from_volume_factor

        self.T.ode = (self.load + air_cooling + power_wall2zone) / C_zone

        # power wall to ambient
        power_wall2amb = (self.T_wall - self.T_ambient) / self.R_hull_amb

        self.T_wall.ode = -(power_wall2amb + power_wall2zone) / self.C_Wall

        # Define ae
        self.T_out.alg = self.T
        self.cooling.alg = air_cooling
        self.power_wall2zone.alg = power_wall2zone

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


def configs() -> List[dict]:
    AGENT_MPC = {
        "id": "myMPCAgent",
        "modules": [
            {"module_id": "Ag1Com", "type": "local_broadcast"},
            # MHE
            {
                "module_id": "mhe",
                "type": "agentlib_mpc.mhe",
                "optimization_backend": {
                    "type": "casadi_mhe",
                    "model": {
                        "type": {"file": __file__, "class_name": RNGRoom.__name__}
                    },
                    "discretization_options": {
                        "collocation_order": 2,
                        "collocation_method": "legendre",
                    },
                    "solver": {"name": "ipopt", "options": {"ipopt.print_level": 0}},
                    "results_file": "results//mhe.csv",
                },
                "horizon": 15,
                "time_step": 200,
                "states": [
                    {"name": "T", "value": 25, "weight": 1},
                    {"name": "T_wall", "value": 27, "weight": 0},
                ],
                "estimated_inputs": [],
                "estimated_parameters": [
                    {"name": "full_capacity_from_volume_factor", "lb": 1, "ub": 10}
                ],
                "known_inputs": [
                    {"name": "mDot", "value": 0.22},
                    {"name": "load", "value": 0},
                    {"name": "T_in", "value": 17},
                    {"name": "T_ambient", "value": 28},
                    {"name": "T_upper", "value": 22},
                ],
                "known_parameters": [
                    # {"name": "full_capacity_from_volume_factor", "value": 5.5},
                ],
            },
            # MPC
            {
                "module_id": "myMPC",
                "type": "agentlib_mpc.mpc",
                "optimization_backend": {
                    "type": "casadi",
                    "model": {
                        "type": {"file": __file__, "class_name": RNGRoom.__name__}
                    },
                    "discretization_options": {
                        "collocation_order": 2,
                        "collocation_method": "legendre",
                    },
                    "solver": {"name": "ipopt", "options": {"ipopt.print_level": 0}},
                    "results_file": "results//mpc.csv",
                },
                "time_step": 200,
                "prediction_horizon": 15,
                "parameters": [
                    {
                        "name": "full_capacity_from_volume_factor",
                        "value": 5.5,
                        "lb": 1,
                        "ub": 10,
                    }
                ],
                "inputs": [
                    {"name": "load", "value": 0},
                    {"name": "T_in", "value": 17},
                    {"name": "T_ambient", "value": 28},
                    {"name": "T_upper", "value": T_UPPER_23},
                ],
                "controls": [{"name": "mDot", "value": 0.02, "ub": 0.1, "lb": 0}],
                "states": [
                    {"name": "T", "value": 25, "ub": 30, "lb": 15},
                    {"name": "T_wall", "value": 27},
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
                    "type": {"file": __file__, "class_name": RNGRoom.__name__},
                    "states": [
                        {"name": "T", "value": 25},
                        {"name": "T_wall", "value": 27},
                    ],
                    "parameters": [
                        {
                            "name": "full_capacity_from_volume_factor",
                            "value": TRUE_CAP_FACTOR,
                        }
                    ],
                },
                "t_sample": 10,
                "update_inputs_on_callback": False,
                "save_results": True,
                "result_causalities": ["input", "output", "local"],
                "outputs": [
                    {"name": "T_out", "value": 25, "alias": "T"},
                ],
                "inputs": [
                    {"name": "load", "value": 0},
                    {"name": "T_in", "value": 17},
                    {"name": "T_ambient", "value": 28},
                    {"name": "T_upper", "value": T_UPPER_23},
                    {"name": "mDot", "value": 0.02, "alias": "mDot"},
                ],
                "states": [
                    {
                        "name": "T_wall",
                        "value": 27,
                        "alias": "T_wall_sim",
                        "shared": False,
                    },
                ],
            },
        ],
    }
    return [AGENT_SIM, AGENT_MPC]


def plots(results):
    fig, ax = plt.subplots(4, 1)

    ax[0].axhline(T_UPPER_23, color="red", label="upper boundary")
    ax[0].plot(results["SimAgent"]["room"]["T_out"], label="temperature")
    room_res = mpc_at_time_step(data=results["myMPCAgent"]["myMPC"], time_step=0)
    ax[0].plot(room_res["variable"]["T"], label="temperature prediction")

    ax[1].plot(results["SimAgent"]["room"]["T_wall"], label="Wall temperature")
    mhe_res = results["myMPCAgent"]["mhe"]
    estimate = last_vals_at_trajectory_index(mhe_res["variable"]["T_wall"].dropna())
    ax[1].plot(estimate, label="Estimated wall temp")

    ax[2].plot(results["SimAgent"]["room"]["mDot"], label="air mass flow")

    ax[3].axhline(TRUE_CAP_FACTOR, label="Capacity Factor")
    estimate = last_vals_at_trajectory_index(
        mhe_res["variable"]["full_capacity_from_volume_factor"].dropna()
    )
    ax[3].plot(estimate, label="Estimated Capacity Factor")

    # for x in ax:
    #     right_legend(x, adjust_right=0.8)
    ax[0].set_ylabel("room temperature / K")
    ax[0].set_xlabel("simulation time / s")
    ax[1].set_ylabel("Wall temperature /°C")
    ax[2].set_ylabel("mass flow / kg/s")
    ax[3].set_ylabel("Capacity Factor")
    ax[1].set_xlabel("simulation time / s")
    ax[2].set_ylim([0, 0.11])
    plt.show()


def main(until: float = 1000, log_level: int = logging.INFO, with_plots: bool = True):
    logging.basicConfig(level=log_level)

    env_config = {"rt": False, "strict": True, "factor": 1, "t_sample": 10}
    mas = LocalMASAgency(
        agent_configs=configs(), env=env_config, variable_logging=False
    )
    mas.run(until=until)
    results = mas.get_results()
    if with_plots:
        plots(results)

    return results


if __name__ == "__main__":
    main(until=3000)
