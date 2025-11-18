"""Example to show interplay of physXAI and agentlib_mpc"""
import logging
import os
import random
import agentlib as al
import matplotlib.pyplot as plt
from agentlib.utils.multi_agent_system import LocalMASAgency


logger = logging.getLogger(__name__)


class InputGeneratorConfig(al.ModelConfig):
    outputs: al.ModelOutputs = [
        al.ModelOutput(
            name="mDot",
            value=0.0225,
            lb=0,
            ub=0.05,
            unit="K",
            description="Air mass flow into zone",
        ),
        # disturbances
        al.ModelOutput(
            name="load",
            value=150,
            lb=150,
            ub=150,
            unit="W",
            description="Heat load into zone",
        ),
        al.ModelOutput(
            name="T_in",
            value=290.15,
            lb=290.15,
            ub=290.15,
            unit="K",
            description="Inflow air temperature",
        ),
    ]


class InputGenerator(al.Model):
    config: InputGeneratorConfig

    def do_step(self, *, t_start, t_sample=None):
        for out in self.config.outputs:
            value = random.random() * (out.ub - out.lb) + out.lb
            self.set(out.name, value)

    def initialize(self, **kwargs):
        pass


def plot(results):
    df = results["Simulator"]["simulator"]
    log = results["PID"]["AgentLogger"]

    fig, (ax_T_out, ax_mDot) = plt.subplots(2, 1, sharex=True)

    (df["T"] - 273.15).plot(ax=ax_T_out, label="Physical", color="black")
    (log["T_set"] - 273.15).plot(ax=ax_T_out, color="black", linestyle="--")

    df["mDot"].plot(ax=ax_mDot, label="mDot", color="black")
    ax_T_out.set_ylabel("$T_{room}$ / °C")
    ax_mDot.set_ylabel("$\dot{m}_{air}$ / kg/s")
    ax_mDot.set_xlabel("Simulation time / s")

    plt.show()


def configs(
    step_size: float = 60,
):
    t_sample_sim = min(max(1, int(step_size) // 30), 10)
    simulator_config = {
        "id": "Simulator",
        "modules": [
            {
                "module_id": "simulator",
                "type": "simulator",
                "model": {
                    "type": {
                        "file": "model.py",
                        "class_name": "PhysicalModel",
                    },
                },
                "t_sample": t_sample_sim,
                "save_results": True,
                "result_filename": "results//simulator.csv",
                "result_causalities": ["local", "input", "output"],
                "overwrite_result_file": True,
                "inputs": [
                    {"name": "mDot", "value": 0.0225, "source": "PID"},
                    {"name": "load", "value": 30},
                    {"name": "T_in", "value": 290.15},
                ],
                "states": [{"name": "T", "shared": True}],
            },
            {
                "module_id": "input_generator",
                "type": "simulator",
                "t_sample": step_size * 10,
                "model": {"type": {"file": __file__, "class_name": "InputGenerator"}},
                "outputs": [
                    {"name": "load", "ub": 150, "lb": 150},
                    {"name": "T_in"},
                ],
            },
            {"type": "local", "subscriptions": ["PID"]},
        ],
    }

    pid_controller = {
        "id": "PID",
        "modules": [
            {
                "module_id": "pid",
                "type": "pid",
                "setpoint": {
                    "name": "setpoint",
                    "value": 273.15 + 22,
                    "alias": "T_set",
                },
                "Kp": 0.01,
                "Ti": 1,
                "input": {"name": "u", "value": 0, "alias": "T"},
                "output": {"name": "y", "value": 0, "alias": "mDot", "shared": "True"},
                "lb": 0,
                "ub": 0.05,
                "reverse": True,
            },
            {
                "module_id": "set_points",
                "type": "agentlib_mpc.set_point_generator",
                "interval": 60 * 10,
                "target_variable": {"name": "T_set", "alias": "T_set"},
            },
            {
                "type": "AgentLogger",
                "values_only": True,
                "t_sample": 300,
                "overwrite_log": True,
            },
            {"type": "local", "subscriptions": ["Simulator"]},
        ],
    }
    return [simulator_config, pid_controller]


def main(
    training_time: float = 3600 * 24 * 0.2,
    plot_results=False,
    step_size: float = 300,
):
    env_config = {"rt": False, "t_sample": 3600}
    logging.basicConfig(level=logging.INFO)
    mas = LocalMASAgency(
        agent_configs=configs(
            step_size=step_size,
        ),
        env=env_config,
        variable_logging=False,
    )
    mas.run(until=training_time + 100)

    ##################################################################################
    # Save training data for physXAI
    results = mas.get_results(cleanup=True)
    df = results["PID"]["AgentLogger"]
    df = df.ffill().bfill()
    os.makedirs("results", exist_ok=True)
    df.to_csv("results//simulation_data.csv")
    ##################################################################################

    if plot_results:
        plot(results)


if __name__ == "__main__":
    main(training_time=3600 * 24 * 1, plot_results=True, step_size=300)
