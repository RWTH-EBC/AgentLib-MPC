

from pathlib import Path
from agentlib_mpc.models.casadi_predictor import CasadiPredictor
from agentlib_mpc.utils.ml_model_eval_plotting import evaluate_model, plot_model_evaluation

class Trainer:
    def __init__(self, save_path):
        self.path = save_path

    def train_models(self, trainer):
        sampled = trainer.resample()
        inputs, outputs = trainer.create_inputs_and_outputs(sampled)
        training_data = trainer.divide_in_tvt(inputs, outputs)

        trainer.fit_ml_model(training_data)
        serialized_ml_model = trainer.serialize_ml_model()
        outputs = training_data.training_outputs.columns
        for name in outputs:
            total_score_mse, metrics_dict = evaluate_model(name, training_data, CasadiPredictor.from_serialized_model(serialized_ml_model))
            train_r2 = metrics_dict[name]["train_score_r2"]

            if abs(1 - train_r2) < abs(1 - best_score):
                best_score = train_r2
                best_serialized_ml_model = serialized_ml_model
                best_metrics = metrics_dict
                best_cross_check = total_score_mse

            if self.trainer_config["save_plots"]:
                self.path.mkdir(parents=True, exist_ok=True)
                plot_model_evaluation(
                    training_data,
                    name,
                    total_score_mse,
                    metrics_dict,
                    CasadiPredictor.from_serialized_model(serialized_ml_model),
                    show_plot=False,
                    save_path=self.path
                )

        best_model_path = Path(self.trainer_config["save_directory"], "best_model", trainer.agent_and_time)
        trainer.save_all(best_serialized_ml_model, training_data, best_model_path, name, best_metrics, best_cross_check)
