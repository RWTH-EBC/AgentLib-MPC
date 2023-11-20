import json
from pathlib import Path
from typing import List
import logging

import numpy as np
import pandas as pd
from agentlib.utils.multi_agent_system import LocalMASAgency
from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
import agentlib as al

# script variables
basepath = Path(__file__).parents[2]
ub = 295.15


def heat_load_func(current):
    """Returns the load on the room in Watt, given a time in seconds."""
    return 100 + 100 * np.sin(np.pi * current / 3600)


class LoadSensorConfig(al.ModelConfig):
    outputs: al.ModelOutputs = [
        al.ModelOutput(name="load_measurement"),
    ]

    parameters: al.ModelParameters = [
        al.ModelParameter(name="uncertainty", value=0.2),
    ]


class LoadSensor(al.Model):
    """Sensor model for output"""

    config: LoadSensorConfig

    def do_step(self, *, t_start, t_sample=None):
        """Returns the current load measurement, with random noise added."""
        random_modifier = 1 + self.uncertainty * (np.random.random() - 0.5)
        d = random_modifier * heat_load_func(t_start)
        self._set_output_value("load_measurement", d)

    def initialize(self, **kwargs):
        pass


class PredictorModuleConfig(al.BaseModuleConfig):
    """Module that outputs a prediction of the heat load at a specified
    interval."""

    outputs: al.AgentVariables = [
        al.AgentVariable(
            name="load_prediction", type="pd.Series", description="test_description"
        ),
        al.AgentVariable(
            name="set_point_schedule", type="pd.Series", description="test_description"
        ),
    ]

    parameters: al.AgentVariables = [
        al.AgentVariable(
            name="sampling_time", value=10, description="Sampling time for prediction."
        ),
        al.AgentVariable(
            name="prediction_length",
            value=10,
            description="Number of sampling points for prediction.",
        ),
        al.AgentVariable(
            name="update_interval",
            value=900,
            description="Time between prediction updates.",
        ),
        al.AgentVariable(
            name="comfort_interval",
            value=7200 * 2,
            description="Time between comfort updates.",
        ),
        al.AgentVariable(
            name="comfort_high",
            value=298,
            description="High value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="comfort_low",
            value=294,
            description="Low value in the comfort set point trajectory.",
        ),
    ]

    shared_variable_fields: list[str] = ["outputs"]


class PredictorModule(al.BaseModule):
    """Module that outputs a prediction of the heat load at a specified
    interval."""

    config: PredictorModuleConfig

    def register_callbacks(self):
        pass

    def process(self):
        """Sets a new prediction at each time step."""
        self.env.process(self.send_comfort_trajectories())

        while True:
            ts = self.get("sampling_time").value
            n = self.get("prediction_length").value
            now = self.env.now
            update_interval = self.get("update_interval").value

            # temperature prediction
            grid = np.arange(now, now + n * ts, ts)
            values = heat_load_func(grid)
            traj = pd.Series(values, index=list(grid))
            self.set("load_prediction", traj)
            yield self.env.timeout(update_interval)

    def send_comfort_trajectories(self):
        """Sends the series for the comfort condition."""
        while True:
            now = self.env.now
            comfort_interval = self.get("comfort_interval").value

            # temperature prediction
            grid = np.arange(now, now + 2 * comfort_interval, 0.5 * comfort_interval)
            values = [self.get("comfort_high").value, self.get("comfort_low").value] * 2
            traj = pd.Series(values, index=list(grid))
            self.set("set_point_schedule", traj)
            yield self.env.timeout(comfort_interval)


class MyCasadiModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        # controls
        CasadiInput(
            name="mDot",
            value=0.0225,
            unit="K",
            description="Air mass flow into zone 0",
        ),
        # disturbances
        CasadiInput(name="d", value=150, unit="W", description="Heat load into zone 0"),
        CasadiInput(
            name="T_in", value=290.15, unit="K", description="Inflow air temperature"
        ),
        # settings
        CasadiInput(
            name="T_set",
            value=294.15,
            unit="K",
            description="Set point for T in objective function",
        ),
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
            name=f"T", value=293.15, unit="K", description="Temperature of zone 0"
        ),
        # slack variables
        CasadiState(
            name=f"T_slack",
            value=0,
            unit="K",
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
            name="c",
            value=100000,
            unit="J/kg*K",
            description="thermal capacity of zone 0",
        ),
        CasadiParameter(
            name="q_T",
            value=1,
            unit="-",
            description="Weight for T in objective function",
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
        CasadiOutput(name="T_out", unit="K", description="Temperature of zone 0")
    ]


class MyCasadiModel(CasadiModel):
    config: MyCasadiModelConfig

    def setup_system(self):
        # Define ode
        self.T.ode = (
            self.cp * self.mDot / self.c * (self.T_in - self.T) + self.d / self.c
        )

        # Define ae
        self.T_out.alg = self.T * 1

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [(0, self.T + self.T_slack, self.T_upper)]

        # Objective function
        objective = sum(
            [
                self.r_mDot * self.mDot,
                self.s_T * self.T_slack**2,
                self.q_T * (self.T - self.T_set) ** 2,
            ]
        )
        return objective


env_config = {"rt": False, "factor": 0.05, "strict": True, "t_sample": 60}
agent_mpc = {
    "id": "myMPCAgent",
    "modules": [
        {"module_id": "Ag1Com", "type": "local_broadcast"},
        {
            "module_id": "myMPC",
            "type": "agentlib_mpc.mpc",
            "optimization_backend": {
                "type": "casadi",
                "model": {"type": {"file": __file__, "class_name": "MyCasadiModel"}},
                "discretization_options": {},
                "results_file": "results//trajectories_mpc.csv",
                "build_batch_bat": "solver_lib/compile_nlp.bat",
                "do_jit": False,
            },
            "time_step": 900,
            "prediction_horizon": 5,
            "parameters": [
                {"name": "q_T", "value": 0},
                {"name": "s_T", "value": 3},
                {"name": "r_mDot", "value": 1},
            ],
            "inputs": [
                {"name": "d", "value": 150, "alias": "load_prediction"},
                {"name": "T_set", "value": 294.55},
                {"name": "T_upper", "value": ub, "interpolation_method": "previous"},
                {"name": "T_in", "value": 290.15},
            ],
            "controls": [{"name": "mDot", "value": 0.02, "ub": 1, "lb": 0}],
            "states": [{"name": "T", "value": 298.16, "ub": 303.15, "lb": 288.15}],
        },
        {"type": "AgentLogger", "t_sample": 3600, "values_only": False},
    ],
}

simulator = {
    "id": "SimAgent",
    "modules": [
        {"module_id": "Ag3Com", "type": "local_broadcast"},
        {
            "module_id": "simulator",
            "type": "simulator",
            "model": {
                "type": {"file": __file__, "class_name": "MyCasadiModel"},
                "states": [{"name": "T", "value": 298.16}],
            },
            "t_sample": 60,
            "save_results": True,
            "inputs": [
                {"name": "mDot", "value": 0.02, "alias": "mDot"},
                {"name": "d", "value": 150, "alias": "load_measurement"},
            ],
            "outputs": [
                {
                    "name": "T_out",
                    "value": 298.16,
                    "ub": 303.15,
                    "lb": 288.15,
                    "alias": "T",
                },
            ],
        },
    ],
}

predictor = {
    "id": "myPredictorAgent",
    "modules": [
        {"module_id": "Ag2Com", "type": "local_broadcast"},
        {
            "module_id": "load_sensor",
            "type": "simulator",
            "model": {
                "type": {"file": __file__, "class_name": "LoadSensor"},
                "parameters": [
                    {"name": "uncertainty", "value": 0.2},
                ],
            },
            "t_sample": 60,
            "outputs": [{"name": "load_measurement", "alias": "load_measurement"}],
        },
        {
            "module_id": "predictor",
            "type": {"file": __file__, "class_name": "PredictorModule"},
            "parameters": [
                {"name": "sampling_time", "value": 60},
                {"name": "prediction_length", "value": 60 * 6},
            ],
            "outputs": [
                {"name": "load_prediction", "value": 60, "shared": True},
                {
                    "name": "set_point_schedule",
                    "value": 60,
                    "alias": "T_upper",
                    "shared": True,
                },
            ],
        },
    ],
}


def run_example(until=3600 * 3, with_plots=True, time_step=0, log_level=logging.INFO):
    """Main function of the mpc_trajectories example."""
    # Set the log-level
    logging.basicConfig(level=log_level)

    mas = LocalMASAgency(
        env=env_config,
        agent_configs=[
            agent_mpc,
            predictor,
            simulator,
        ],
    )
    mas.run(until=until)

    results = mas.get_results()
    if with_plots:
        import matplotlib.pyplot as plt
        from agentlib_mpc.utils.plotting import basic
        from agentlib_mpc.utils.plotting.mpc import plot_mpc

        mpc_results = results["myMPCAgent"]["myMPC"]
        sim_res = results["SimAgent"]["simulator"]
        sim_res.index = sim_res.index / 3600
        controller = results["myMPCAgent"]["AgentLogger"]
        controller.set_index(controller.index / 3600, inplace=True)

        set_points = controller["T_upper"].iloc[0].value - 273.15
        set_points.index = set_points.index.astype(np.float64) / 3600
        # set_points2 = controller["T_upper"].loc[7260 / 3600].value - 273.15
        # set_points2.index = set_points2.index.astype(np.float64) / 3600

        fig, ax = plt.subplots(3, 1, sharex=True)

        # temperature
        plot_mpc(
            mpc_results["variable"]["T"] - 273.15,
            plot_predictions=False,
            plot_actual_values=True,
            ax=ax[0],
            convert_to="hours",
        )

        # controller.plot(y="T", ax=ax[0], label="temperature")
        # t0 = mpc_at_time_step(data=mpc_results, time_step=time_step, variable="T")
        # t0.plot(y="variable", ax=ax[0], ls="--", label="prediction")
        set_points.plot(
            ax=ax[0],
            color="red",
            drawstyle="steps-post",
            label="_nolegend_",
        )
        # set_points2.plot(
        #     ax=ax[0],
        #     color="red",
        #     drawstyle="steps-post",
        #     label="_nolegend_",
        # )
        # ax[0].axhline(ub, color="red", label="upper boundary")

        # air mass flow
        sim_res.plot(y="mDot", color="0", ax=ax[1], label="_nolegend_")

        # actual load
        sim_res.plot(y="d", color="0", ax=ax[2], label="_nolegend_")

        ax[0].set_ylabel(r"$T_{room}$ / °C")
        ax[1].set_ylabel(r"$\dot{m}_{air}$ / $\frac{kg}{s}$")
        ax[2].set_ylabel(r"$\dot{Q}_{gains}$ / W")
        ax[2].set_xlabel("Time / h")

        ax[1].get_legend().remove()
        ax[2].get_legend().remove()

        for x in ax:

            basic.make_grid(x)

        plt.show()

    return results


if __name__ == "__main__":
    run_example(until=3600 * 6, time_step=8000)
