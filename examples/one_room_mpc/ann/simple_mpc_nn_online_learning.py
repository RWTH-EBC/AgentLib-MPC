import logging
from pathlib import Path
import os
import agentlib as al
import random

from agentlib.utils.multi_agent_system import LocalMASAgency

logger = logging.getLogger(__name__)

# script variables
ub = 295.15

ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 3600, "offset": 0}
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

def agent_configs(ml_model_path: str) -> list[dict]:
    agent_mpc = {
        "id": "myMPCAgent",
        "modules": [
            {"module_id": "Ag1Com", "type": "local_broadcast"},
            {
                "module_id": "myMPC",
                "type": "agentlib_mpc.mpc",
                "optimization_backend": {
                    "type": "casadi_ml",
                    "model": {
                        "type": {
                            "file": "model.py",
                            "class_name": "DataDrivenModel",
                        },
                        "ml_model_sources": [ml_model_path],
                    },
                    "discretization_options": {
                        "method": "multiple_shooting",
                    },
                    "results_file": "results//opt.csv",
                    "overwrite_result_file": True,
                    "solver": {"name": "ipopt", "options": {"ipopt.print_level": 0}},
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
                "states": [{"name": "T", "value": 298.16, "ub": 303.15, "lb": 288.15}],
            },
            {
                "type": "local",
                "subscriptions": [
                    "Trainer_OL"
                ]
            }

        ],
    }
    agent_sim = {
        "id": "SimAgent",
        "modules": [
            {"module_id": "Ag1Com", "type": "local_broadcast"},
            {
                "module_id": "room",
                "type": "simulator",
                "model": {
                    "type": {
                        "file": "model.py",
                        "class_name": "PhysicalModel",
                    },
                    "states": [{"name": "T", "value": 298.16}],
                },
                "t_sample": 10,
                "save_results": True,
                "update_inputs_on_callback": False,
                "overwrite_result_file": True,
                "result_filename": "results//simulation_data_onlinelearning.csv",
                "outputs": [
                    {"name": "T_out", "value": 298, "alias": "T"},
                    {"name": "T_in_sim", "value": 290.15},
                    {"name": "load_sim", "value": 150},
                ],
                "inputs": [
                    {"name": "mDot", "value": 0.02, "alias": "mDot"},
                ],
            },
            {
                "type": "local",
                "subscriptions": [
                    "Trainer_OL"
                ]
            },
        ],
    }
    return [agent_mpc, agent_sim]


def run_example(with_plots=True, log_level=logging.INFO, until=8000, initial_training_time=43200, retrain_delay=0):
    # Change the working directly so that relative paths work
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    logging.basicConfig(level=log_level)

    try:
        ann_path = list(Path.cwd().glob("anns/best_model/*/ml_model.json"))[-1]
    except IndexError:
        import training_ann
        training_ann.main(training_time=3600 * 24 * 1, plot_results=False, step_size=300)
        ann_path = list(Path.cwd().glob("anns/best_model/ml_model.json"))[-1]

    #Run with Online Learning
    agent_config = agent_configs(ml_model_path=str(ann_path))
    import training_nn_onlinelearning
    trainer = training_nn_onlinelearning.get_trainer(retrain_delay, ann_path)
    agent_config.append(trainer)

    mas = LocalMASAgency(
        agent_configs=agent_config,
        env=ENV_CONFIG,
        variable_logging=False,
    )
    mas.run(until=until)
    results = mas.get_results()
    if with_plots:
        import matplotlib.pyplot as plt
        from agentlib_mpc.utils.plotting.mpc import plot_mpc

        mpc_results = results["myMPCAgent"]["myMPC"]
        sim_res = results["SimAgent"]["room"]
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

        fig, ax = plt.subplots(2, 1, sharex=True)
        temperature = mpc_results["variable"]["T"] - 273.15
        plot_mpc(
            series=temperature,
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


if __name__ == "__main__":
    run_example(until=86400*3, retrain_delay=86400)
