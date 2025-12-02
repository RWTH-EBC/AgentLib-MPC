from typing import Union

import numpy as np
import pandas as pd


from agentlib_mpc.models.casadi_predictor import CasadiPredictor, casadi_predictors
from agentlib_mpc.models.serialized_ml_model import SerializedMLModel
from agentlib_mpc.data_structures import ml_model_datatypes


def predict_array(
    df: pd.DataFrame, ml_model: CasadiPredictor, outputs: pd.Index
) -> pd.DataFrame:
    arr = (
        ml_model.predict(df.values.reshape(1, -1))
        .toarray()
        .reshape((df.shape[0], len(outputs)))
    )
    return pd.DataFrame(arr, columns=outputs, index=df.index)






def evaluate_model(
        name,
        training_data: ml_model_datatypes.TrainingData,
        model: Union[CasadiPredictor, SerializedMLModel]
) -> tuple[float, dict]:
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


