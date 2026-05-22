from typing import Any


def get_config() -> dict[str, Any]:

    return {
        "id": "myMPCAgent",
        "modules": [
            {"module_id": "Ag1Com", "type": "local_broadcast"},
            {
                "module_id": "myMPC",
                "type": "agentlib_mpc.mpc",
                "optimization_backend": {
                    "type": "casadi",
                    "model": {
                        "type": {
                            "file": "mpc/part3_simple_mpc_model.py",
                            "class_name": "SimpleRoom",
                        }
                    },
                    "results_file": "results/mpc_simple_building_local_broadcast.csv",
                    "overwrite_result_file": True,

                    # Solver settings and discretization options can be specified here. 
                    # The default method for discretization is direct collocation. To override this, you can set the "method" key to "multiple_shooting".
                    "discretization_options": {
                        # "method": "multiple_shooting",
                        "collocation_order": 1,                 
                        "collocation_method": "radau",  # possible alternative: "legendre"
                    },
                    "solver": {
                        "name": "ipopt",                            # possible alternative: "fatrop". Fields set in "options" remain the same
                        "options": {
                            "ipopt": {
                                "max_iter": 30,                     # maximum iterations of the solver
                                "tol": 0.00001,                     # tolerance used by the solver to determine an optimum
                                "acceptable_tol": 1,                # tolerance used by the solver to determine an acceptable solution, then marked as "solved to acceptable level"
                                "acceptable_constr_viol_tol": 1,
                                "acceptable_iter": 3,
                                "acceptable_compl_inf_tol": 1,
                                "print_level": 0,                   # level to determine the level of detail for the solver output, level 5 includes extensive info on the optimization problem
                            }
                        },
                    },
                },
                "time_step": 900,
                "prediction_horizon": 8,
                "parameters": [
                    {"name": "s_T", "value": 1000},
                    {"name": "r_pel", "value": 0.1},
                ],
                "inputs": [
                    {
                        "name": "T_upper",
                        "value": 299,
                        "interpolation_method": "previous",
                    },
                    {
                        "name": "T_lower",
                        "value": 290,
                        "interpolation_method": "previous",
                    },
                    {"name": "T_amb", "value": 278.15},
                ],
                "controls": [{"name": "Q_in", "value": 100, "ub": 200, "lb": -100}],
                "states": [
                    {"name": "T_zone", "value": 294.15, "ub": 303.15, "lb": 288.15}
                ],
                "outputs": [{"name": "P_el", "value": 0}],
            },
        ],
    }
