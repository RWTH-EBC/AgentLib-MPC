from typing import Union

import casadi as ca
import numpy as np

from agentlib_mpc.models.casadi_predictor.casadi_predictor import CasadiPredictor
from agentlib_mpc.models.serialized_ml_model.serialized_gpr import SerializedGPR


class CasadiGPR(CasadiPredictor):
    """
    Generic implementation of scikit-learn Gaussian Process Regressor.
    """

    def __init__(self, serialized_model: SerializedGPR) -> None:
        super().__init__(serialized_model)

    @property
    def input_shape(self) -> tuple[int, int]:
        """Input shape of Predictor."""
        return 1, self.predictor_model.X_train_.shape[1]

    def _build_prediction_function(self) -> ca.Function:
        """Build the prediction function with casadi and a symbolic input."""
        normalize = self.predictor_model.data_handling.normalize
        scale = self.predictor_model.data_handling.scale
        alpha = self.predictor_model.alpha_
        if normalize:
            normalized_inp = self._normalize(self.sym_input)
            k_star = self._kernel(normalized_inp)
        else:
            k_star = self._kernel(self.sym_input)
        f_mean = ca.mtimes(k_star.T, alpha) * scale
        return ca.Function("forward", [self.sym_input], [f_mean])

    def _kernel(
        self,
        x_test: ca.MX,
    ) -> ca.MX:
        """
        Calculates the kernel with regard to mpc and testing data.
        If x_train is None the internal mpc data is used.

        shape(x_test)  = (n_samples, n_features)
        shape(x_train) = (n_samples, n_features)
        """

        square_distance = self._square_distance(x_test)
        length_scale = self.predictor_model.kernel_.k1.k2.length_scale
        constant_value = self.predictor_model.kernel_.k1.k1.constant_value
        return np.exp((-square_distance / (2 * length_scale**2))) * constant_value

    def _square_distance(self, inp: ca.MX):
        """
        Calculates the square distance from x_train to x_test.

        shape(x_test)  = (n_test_samples, n_features)
        shape(x_train) = (n_train_samples, n_features)
        """

        x_train = self.predictor_model.X_train_

        self._check_shapes(inp, x_train)

        a = ca.sum2(inp**2)

        b = ca.np.sum(x_train**2, axis=1, dtype=float).reshape(-1, 1)

        c = -2 * ca.mtimes(x_train, inp.T)

        return a + b + c

    def _normalize(self, x: ca.MX):
        mean = self.predictor_model.data_handling.mean
        std = self.predictor_model.data_handling.std

        if mean is None and std is not None:
            raise ValueError("Mean and std are not valid.")

        return (x - ca.DM(mean).T) / ca.DM(std).T

    def _check_shapes(self, x_test: Union[ca.MX, np.ndarray], x_train: np.ndarray):
        if x_test.shape[1] != x_train.shape[1]:
            raise ValueError(
                f"The shape of x_test {x_test.shape}[1] and x_train {x_train.shape}[1] must match."
            )
