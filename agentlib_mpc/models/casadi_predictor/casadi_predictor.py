import abc

import casadi as ca
import numpy as np

from typing import Union, TYPE_CHECKING, Type

from agentlib_mpc.models.serialized_ml_model import (
    SerializedMLModel,
    MLModels,
)

if TYPE_CHECKING:
    from keras import Sequential
    from agentlib_mpc.models.serialized_ml_model import CustomGPR
    from sklearn.linear_model import LinearRegression


class CasadiPredictor(abc.ABC):
    """
    Protocol for generic Casadi implementation of various ML-Model-based predictors.

    Attributes:
        serialized_model: Serialized model which will be translated to a casadi model.
        predictor_model: Predictor model from other libraries, which are translated to
        casadi syntax.
        sym_input: Symbolical input of predictor. Has the necessary shape of the input.
        prediction_function: Symbolical casadi prediction function of the given model.
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, serialized_model: SerializedMLModel) -> None:
        """Initialize Predictor class."""
        self.serialized_model: SerializedMLModel = serialized_model
        self.predictor_model: Union[
            Sequential, CustomGPR, LinearRegression
        ] = serialized_model.deserialize()
        self.sym_input: ca.MX = self._get_sym_input()
        self.prediction_function: ca.Function = self._build_prediction_function()

    @classmethod
    def from_serialized_model(cls, serialized_model: SerializedMLModel):
        """Initialize sub predictor class."""
        model_type = serialized_model.model_type
        # todo return type[cls]
        return casadi_predictors(model_type)(serialized_model)

    @property
    @abc.abstractmethod
    def input_shape(self) -> tuple[int, int]:
        """Input shape of Predictor."""
        pass

    @property
    def output_shape(self) -> tuple[int, int]:
        """Output shape of Predictor."""
        return 1, len(self.serialized_model.output)

    def _get_sym_input(self):
        """Returns symbolical input object in the required shape."""
        return ca.MX.sym("input", 1, self.input_shape[1])

    @abc.abstractmethod
    def _build_prediction_function(self) -> ca.Function:
        """Build the prediction function with casadi and a symbolic input."""
        pass

    def predict(self, x: Union[np.ndarray, ca.MX]) -> Union[ca.DM, ca.MX]:
        """
        Evaluate prediction function with input data.
        Args:
            x: input data.
        Returns:
            results of evaluation of prediction function with input data.
        """
        return self.prediction_function(x)


def casadi_predictors(model_type: MLModels) -> Type[CasadiPredictor]:
    if model_type == MLModels.ANN:
        from agentlib_mpc.models.casadi_predictor.casadi_ann import CasadiANN

        return CasadiANN
    if model_type == MLModels.GPR:
        from agentlib_mpc.models.casadi_predictor.casadi_gpr import CasadiGPR

        return CasadiGPR
    if model_type == MLModels.LINREG:
        from agentlib_mpc.models.casadi_predictor.casadi_linreg import CasadiLinReg

        return CasadiLinReg
