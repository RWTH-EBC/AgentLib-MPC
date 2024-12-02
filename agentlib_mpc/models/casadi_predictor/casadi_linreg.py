import casadi as ca

from agentlib_mpc.models.casadi_predictor.casadi_predictor import CasadiPredictor
from agentlib_mpc.models.serialized_ml_model.serialized_linreg import SerializedLinReg


class CasadiLinReg(CasadiPredictor):
    """
    Generic Casadi implementation of scikit-learn LinerRegression.
    """

    def __init__(self, serialized_model: SerializedLinReg) -> None:
        """
        Initializes CasadiLinReg predictor.
        Args:
            serialized_model: SerializedLinReg object.
        """
        super().__init__(serialized_model)

    @property
    def input_shape(self) -> tuple[int, int]:
        """Input shape of Predictor."""
        return 1, self.predictor_model.coef_.shape[1]

    def _build_prediction_function(self) -> ca.Function:
        """Build the prediction function with casadi and a symbolic input."""
        intercept = self.predictor_model.intercept_
        coef = self.predictor_model.coef_
        function = intercept + ca.mtimes(self.sym_input, coef.T)
        return ca.Function("forward", [self.sym_input], [function])
