import logging
import sys
from pathlib import Path
import os
from agentlib.core.errors import OptionalDependencyError
from agentlib.utils.multi_agent_system import LocalMASAgency
from agentlib_mpc.machine_learning_plugins.physXAI.model_generation import generate_physxai_model


logger = logging.getLogger(__name__)


# script variables
ub = 295.15


ENV_CONFIG = {"rt": False, "factor": 0.01, "t_sample": 60}


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
                "outputs": [
                    {"name": "T_out", "value": 298, "alias": "T"},
                ],
                "inputs": [
                    {"name": "mDot", "value": 0.02, "alias": "mDot"},
                ],
            },
        ],
    }
    return [agent_mpc, agent_sim]


def run_example(with_plots=True, log_level=logging.INFO, until=8000, testing=False):
    ##################################################################################
    # Import example training script of physXAI
    try:
        import physXAI.agentlib_mpc_plugin as physXAI_plugin
    except ImportError:
        raise OptionalDependencyError(dependency_name="physXAI", dependency_install="git+https://github.com/RWTH-EBC/physXAI.git", used_object="physXAI")
    ##################################################################################

    script_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(script_dir)

    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    logging.basicConfig(level=log_level)

    ##################################################################################
    # Generate training data
    import generate_train_data
    if testing:
        generate_train_data.main(
            training_time=3600 * 2,  # Much shorter training time
            plot_results=False,
            step_size=300,
        )
    else:
        if not Path("results//simulation_data.csv").exists():
            generate_train_data.main(training_time=3600 * 24 * 1, plot_results=False, step_size=300)
    ##################################################################################

    ##################################################################################
    # Main interface between physXAI and agentlib_mpc to generate the ML model
    files = generate_physxai_model(
        models=['example'],  # Call example model
        physXAI_scripts_path=os.path.dirname(physXAI_plugin.__file__), # Get path of physXAI example scripts
        training_data_path='results//simulation_data.csv', # Generated training data
        run_id='001', # Model ID
        time_step=300  # Synchronize with MPC time step
    )
    ##################################################################################

    mas = LocalMASAgency(
        agent_configs=agent_configs(ml_model_path=files[0]),  # Paste generated model path (only one model here)
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
    run_example(with_plots=True, until=3600)
