
def get_trainer(retrain_delay, ml_model_source=None):
    trainer_config = {
        "id": "Trainer_OL",
        "modules": [
            {
                "step_size": 300,
                "module_id": "trainer",
                "type": "agentlib_mpc.ann_trainer",
                "epochs": 1000,
                "batch_size": 128,
                "online_learning": {
                    "active": True,
                    "training_at": retrain_delay,
                    "initial_ml_model_path": ml_model_source
                },
                "inputs": [
                    {"name": "mDot", "value": 0.0225, "source": "myMPCAgent"},
                    {"name": "load", "alias": "load_sim", "value": 30, "source": "SimAgent"},
                    {"name": "T_in", "alias": "T_in_sim", "value": 290.15, "source": "SimAgent"},
                ],
                "outputs": [{"name": "T", "value": 273.15 + 22}],
                # the lags here are not needed, but we have them to validate the code
                "lags": {"load": 2, "T": 2, "mDot": 3},
                "output_types": {"T": "difference"},
                "interpolations": {"mDot": "mean_over_interval"},
                "layers": [{32, "sigmoid"}],
                "train_share": 0.6,
                "validation_share": 0.2,
                "test_share": 0.2,
                "save_directory": "anns",
                "use_values_for_incomplete_data": True,
                "data_sources": ["results//simulation_data.csv"],
                "save_data": True,
                "save_ml_model": True,
                "save_plots": True,
                "early_stopping": {"activate": "True", "patience": 800},
            },
            {"type": "local", "subscriptions": ["SimAgent", "myMPCAgent"]}
        ]
    }

    return trainer_config


