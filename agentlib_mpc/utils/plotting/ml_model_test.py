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
        copy_empty[i : i + len(pred)] = pred
        i += len(pred)

        copy_empty = copy_empty[idx]

        out.append(copy_empty)

    return out, true_sorted


# Change
def evaluate_model(
    training_data: ml_model_datatypes.TrainingData,
    model: Union[CasadiPredictor, SerializedMLModel],
    metric: Callable = None,
    show_plot: bool = True,
    save_path: Optional[Path] = None,
):
    """Tests the Model on test data"""

    if metric is None:
        metric = lambda x: x * x

    # make model executable
    if isinstance(model, SerializedMLModel):
        model_ = casadi_predictors[model.model_type](model)
    else:
        model_ = model

    # # make the predictions
    outputs = training_data.training_outputs.columns

    train_pred = predict_array(
        df=training_data.training_inputs, ml_model=model_, outputs=outputs
    )
    valid_pred = predict_array(
        df=training_data.validation_inputs, ml_model=model_, outputs=outputs
    )
    test_pred = predict_array(
        df=training_data.test_inputs, ml_model=model_, outputs=outputs
    )
    train_error = training_data.training_outputs - train_pred
    valid_error = training_data.validation_outputs - valid_pred
    test_error = training_data.test_outputs - test_pred

    for name in outputs:
        train_score = calc_scores(train_error[name], metric=metric)
        valid_score = calc_scores(valid_error[name], metric=metric)
        test_score = calc_scores(test_error[name], metric=metric)
        total_score = sum([train_score, valid_score, test_score]) / 3

        # plot
        y_pred_sorted, y_true_sorted = pairwise_sort(
            (training_data.training_outputs[name].values, train_pred[name]),
            (training_data.validation_outputs[name].values, valid_pred[name]),
            (training_data.test_outputs[name].values, test_pred[name]),
        )

        scale = range(len(y_true_sorted))

        with basic.Style() as style:
            fig, ax = basic.make_fig(style=style)
            ax: plt.Axes
            for y, c, label in zip(
                y_pred_sorted,
                [basic.EBCColors.red, basic.EBCColors.green, basic.EBCColors.blue],
                ["Train", "Valid", "Test"],
            ):
                if not all(np.isnan(y)):
                    ax.scatter(scale, y, s=0.6, color=c, label=label)

            ax.scatter(
                scale,
                y_true_sorted,
                s=0.6,
                color=basic.EBCColors.dark_grey,
                label="True",
            )
            ax.set_xlabel("Samples")
            ax.legend(loc="upper left")
            ax.yaxis.grid(linestyle="dotted")
            ax.set_title(
                f"{name}\ntest_score={test_score.__round__(4)}\ntotal_score={total_score.__round__(4)}"
            )
            if show_plot:
                fig.show()
            if save_path is not None:
                fig.savefig(fname=Path(save_path, f"evaluation_{name}.png"))
