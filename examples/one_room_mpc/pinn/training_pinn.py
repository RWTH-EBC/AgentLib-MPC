import logging
from agentlib.utils.multi_agent_system import LocalMASAgency
import os
import sys
import agentlib as al
import random
import matplotlib.pyplot as plt

#todo: check ist sys.path.append() notwendig?
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model

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


def configs(training_time, step_size, plot_results):
    trainer_config = {
        "id": "Trainer",
        "modules": [
            {
                "step_size": step_size,
                "module_id": "trainer_testing_new",
                "type": "agentlib_mpc.pinn_trainer",
                "epochs": 1000,
                "batch_size": 64,
                "loss_function": "mean_squared_error",
                "number_of_training_repetitions": 2,
                "weight_phys_losses": 0.5,
                "additional_losses": [
                    {
                        "name": "PhysicalLoss",
                        "scale": 1,
                        "function": "T_pred",
                        "features": ["T"],
                        "module_path": "physical_eq",
                    },
                ],
                "inputs": [
                    {"name": "mDot", "value": 0.0225, "source": "PID"},
                    {"name": "load", "value": 30, "source": "Simulator"},
                    {"name": "T_in", "value": 290.15, "source": "Simulator"},
                ],
                "outputs": [{"name": "T", "value": 273.15 + 22}],
                "lags": {"load": 2, "T": 2, "mDot": 3},
                "output_types": {"T": "difference"},
                "interpolations": {"mDot": "mean_over_interval"},
                "layers": [{32, "sigmoid"}],
                "train_share": 0.6,
                "validation_share": 0.2,
                "test_share": 0.2,
                "retrain_delay": training_time,
                "save_directory": "pinns",
                "use_values_for_incomplete_data": True,
                "data_sources": ["results//simulation_data.csv"],
                "save_data": True,
                "save_ml_model": True,
                "save_plots": True,
                "early_stopping": {"activate": "False", "patience": 800},
            },
            {"type": "local", "subscriptions": ["Simulator", "PID"]},
        ],
    }
    t_sample_sim = min(max(1, int(step_size) // 30), 10)
    simulator_config = {
        "id": "Simulator",
        "modules": [
            {
                "module_id": "simulator",
                "type": "simulator",
                "model": {
                    "type": {
                        "file": model.__file__,
                        "class_name": model.PhysicalModel.__name__,
                    },
                },
                "t_sample": t_sample_sim,
                "save_results": plot_results,
                "result_filename": "results//simulation_data.csv",
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
                    # {"name": "mDot"},
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
            {"type": "AgentLogger", "values_only": True, "t_sample": 3600},
            {"type": "local", "subscriptions": ["Simulator"]},
        ],
    }
    return [simulator_config, trainer_config, pid_controller]

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
def main(days, training_time: float = 1000, step_size: float = 300):
    env_config = {"rt": False, "t_sample": 3600}
    logging.basicConfig(level=logging.INFO)
    mas = LocalMASAgency(
        agent_configs=configs(training_time, step_size, plot_results=True),
        env=env_config,
        variable_logging=False,
    )

    mas.run(until=training_time + 100)
    results = mas.get_results(cleanup=True)
    plot(results)

    return results


if __name__ == "__main__":
    step_size = 300
    days = 1
    main(days=days, training_time=86400 * days, step_size=step_size)
