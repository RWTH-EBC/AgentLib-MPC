from pathlib import Path
from typing import Callable, Union, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from agentlib_mpc.models.casadi_predictor import CasadiPredictor, casadi_predictors
from agentlib_mpc.models.serialized_ml_model import SerializedMLModel
from agentlib_mpc.utils.plotting import basic
from agentlib_mpc.data_structures import ml_model_datatypes


def calc_scores(errors: np.ndarray, metric: Callable) -> float:
    if all(np.isnan(errors)):
        return 0

    return float(np.mean(metric(errors)))


def predict_array(
    df: pd.DataFrame, ml_model: CasadiPredictor, outputs: pd.Index
) -> pd.DataFrame:
    arr = (
        ml_model.predict(df.values.reshape(1, -1))
        .toarray()
        .reshape((df.shape[0], len(outputs)))
    )
    return pd.DataFrame(arr, columns=outputs, index=df.index)


def pairwise_sort(*arrays: tuple[np.ndarray, np.ndarray]):
    true_sorted = np.concatenate([true.flatten() for true, pred in arrays])
    empty = np.empty(shape=true_sorted.shape)
    empty[:] = np.nan

    idx = np.argsort(true_sorted)
    true_sorted = true_sorted[idx]

    i = 0
    out = list()

    for _, pred in arrays:
        copy_empty = empty.copy()
        copy_empty[i: i + len(pred)] = pred
        i += len(pred)

        copy_empty = copy_empty[idx]

        out.append(copy_empty)

    return out, true_sorted



def evaluate_model(
        name,
        training_data: ml_model_datatypes.TrainingData,
        model: Union[CasadiPredictor, SerializedMLModel]
) -> tuple[float, dict, float]:
    """Evaluates the Model and returns primary score, metrics dict, and cross-check score"""

    if isinstance(model, SerializedMLModel):
        model_ = casadi_predictors[model.model_type](model)
    else:
        model_ = model

    # make the predictions
    outputs = training_data.training_outputs.columns

    train_pred = predict_array(
        df=training_data.training_inputs, ml_model=model_, outputs=outputs
    )
    valid_pred = predict_array(
        df=training_data.validation_inputs, ml_model=model_, outputs=outputs
    )
    test_pred = predict_array(df=training_data.test_inputs, ml_model=model_, outputs=outputs)

    train_true = training_data.training_outputs[name].values
    valid_true = training_data.validation_outputs[name].values
    test_true = training_data.test_outputs[name].values

    metrics_dict = {}
    sum_total_score = 0

    metrics_dict[name] = {}

    train_score_mse = np.mean((train_true - train_pred[name]) ** 2)
    valid_score_mse = np.mean((valid_true - valid_pred[name]) ** 2)
    test_score_mse = np.mean((test_true - test_pred[name]) ** 2)

    def calculate_r2(y_true, y_pred):
        """Calculate R² score"""
        if len(y_true) == 0 or len(y_pred) == 0:
            return float('nan')
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

    train_r2 = calculate_r2(train_true, train_pred[name])
    valid_r2 = calculate_r2(training_data.validation_outputs[name].values, valid_pred[name]) if hasattr(training_data,'validation_outputs') else float('nan')
    test_r2 = calculate_r2(training_data.test_outputs[name].values, test_pred[name]) if hasattr(training_data,'test_outputs') else float('nan')

    total_score_mse = (train_score_mse + valid_score_mse + test_score_mse) / 3

    metrics_dict[name] = {
        'train_score_mse': train_score_mse,
        'valid_score_mse': valid_score_mse,
        'test_score_mse': test_score_mse,
        'train_score_r2': train_r2,
        'valid_score_r2': valid_r2,
        'test_score_r2': test_r2
    }

    return total_score_mse, metrics_dict


def plot_model_evaluation(
        training_data: ml_model_datatypes.TrainingData,
        name,
        cross_check_score: float,
        metrics_dict: dict,
        model: Union[CasadiPredictor, SerializedMLModel],
        show_plot: bool = True,
        save_path: Optional[Path] = None,
):
    """Plots the Model evaluation on test data"""
    model_ = model if not isinstance(model, SerializedMLModel) else casadi_predictors[model.model_type](model)
    outputs = training_data.training_outputs.columns

    train_pred = predict_array(
        df=training_data.training_inputs, ml_model=model_, outputs=outputs
    )
    valid_pred = predict_array(
        df=training_data.validation_inputs, ml_model=model_, outputs=outputs
    )
    test_pred = predict_array(df=training_data.test_inputs, ml_model=model_, outputs=outputs)

    train_true = training_data.training_outputs[name].values
    valid_true = training_data.validation_outputs[name].values
    test_true = training_data.test_outputs[name].values


    with basic.Style() as style:
        fig, ax = basic.make_fig(style=style)

        # First subplot: Time series plot
        y_pred_sorted, y_true_sorted = pairwise_sort(
            (train_true, train_pred[name]),
            (valid_true, valid_pred[name]),
            (test_true, test_pred[name]),
        )

        scale = range(len(y_true_sorted))

        for y, c, label in zip(
                y_pred_sorted,
                [basic.EBCColors.red, basic.EBCColors.green, basic.EBCColors.blue],
                ["Train", "Valid", "Test"],
        ):
            if not all(np.isnan(y)):
                ax.scatter(scale, y, s=0.6, color=c, label=label)

        ax.scatter(scale, y_true_sorted, s=0.6, color=basic.EBCColors.dark_grey, label="True")
        ax.set_xlabel("Samples")
        ax.legend(loc="upper left")
        ax.yaxis.grid(linestyle="dotted")
        ax.set_title(
            f"{name}\ntotal_score={cross_check_score.__round__(4)}")

        plt.tight_layout()
        show_plot = True
        if show_plot:
            plt.show()
        if save_path is not None:
            fig.savefig(fname=Path(save_path, f"evaluation_mse_{name}.png"))

        with basic.Style() as style:
            fig, ax = basic.make_fig(style=style)

            # Get overall min and max for axis limits with some padding
            all_values = np.concatenate([
                train_true, train_pred[name],
                training_data.validation_outputs[name].values if hasattr(training_data, 'validation_outputs') else [],
                valid_pred[name] if hasattr(training_data, 'validation_outputs') else [],
                training_data.test_outputs[name].values if hasattr(training_data, 'test_outputs') else [],
                test_pred[name] if hasattr(training_data, 'test_outputs') else []
            ])

            min_val = np.min(all_values)
            max_val = np.max(all_values)
            padding = (max_val - min_val) * 0.1
            plot_min = min_val - padding
            plot_max = max_val + padding

            # Plot angle bisector (True=Predicted line)
            ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', label='Perfect Prediction')

            # Plot data points
            ax.scatter(train_true, train_pred[name],
                       color=basic.EBCColors.red, label=f'Train (R²={metrics_dict[name]["train_score_r2"]:.3f})', alpha=0.6, s=20)

            if hasattr(training_data, 'validation_outputs') and len(training_data.validation_outputs) > 0:
                valid_true = training_data.validation_outputs[name].values
                ax.scatter(valid_true, valid_pred[name],
                           color=basic.EBCColors.green, label=f'Valid (R²={metrics_dict[name]["valid_score_r2"]:.3f})', alpha=0.6, s=20)

            if hasattr(training_data, 'test_outputs') and len(training_data.test_outputs) > 0:
                test_true = training_data.test_outputs[name].values
                ax.scatter(test_true, test_pred[name],
                           color=basic.EBCColors.blue, label=f'Test (R²={metrics_dict[name]["test_score_r2"]:.3f})', alpha=0.6, s=20)

            # Set labels and title
            ax.set_xlabel('True Value')
            ax.set_ylabel('Predicted Value')
            ax.set_title(f'{name}')

            # Set equal axis limits with padding
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)

            # Move legend outside
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Add grid
            ax.grid(linestyle='dotted')

            # Make plot square
            ax.set_aspect('equal')

            # Adjust layout to prevent legend overlap
            plt.tight_layout()

            if show_plot:
                plt.show()
            if save_path is not None:
                fig.savefig(fname=Path(save_path, f"evaluation_scatter_{name}.png"),
                            bbox_inches='tight')  # Ensure legend is included in saved figure

