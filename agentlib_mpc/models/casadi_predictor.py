import abc

import casadi as ca
import numpy as np

from aenum import Enum
from keras import layers

from typing import Union, TYPE_CHECKING

from agentlib_mpc.models.serialized_ml_model import (
    SerializedMLModel,
    SerializedLinReg,
    SerializedGPR,
    SerializedANN,
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
        return casadi_predictors[model_type](serialized_model)

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


###################################
###             ANN             ###
###################################


class ANNLayerTypes(str, Enum):
    DENSE = "dense"
    FLATTEN = "flatten"
    BATCHNORMALIZATION = "batch_normalization"
    LSTM = "lstm"
    RESCALING = "rescaling"


class Layer:
    """
    Single layer of an artificial neural network.
    """

    def __init__(self, layer: layers.Layer):
        self.config = layer.get_config()

        # name
        if "name" in self.config:
            self.name = self.config["name"]

        # units
        if "units" in self.config:
            self.units = self.config["units"]

        # activation function
        if "activation" in self.config:
            self.activation = self.get_activation(layer.get_config()["activation"])

        # input / output shape
        self.input_shape = layer.input_shape[1:]
        self.output_shape = layer.output_shape[1:]

        # update the dimensions to two dimensions
        self.update_dimensions()

        # symbolic input layer
        self.input_layer = ca.MX.sym(
            "input_layer", self.input_shape[0], self.input_shape[1]
        )

    def __str__(self):
        ret = ""

        if hasattr(self, "units"):
            ret += f"\tunits:\t\t\t\t{self.units}\n"
        if hasattr(self, "activation"):
            ret += f"\tactivation:\t\t\t{self.activation.__str__()}\n"
        if hasattr(self, "recurrent_activation"):
            ret += f"\trec_activation:\t\t{self.recurrent_activation.__str__()}\n"
        ret += f"\tinput_shape:\t\t{self.input_shape}\n"
        ret += f"\toutput_shape:\t\t{self.output_shape}\n"

        return ret

    def update_dimensions(self):
        """
        CasADi does only work with two dimensional arrays. So the dimensions must be updated.
        """

        if len(self.input_shape) == 1:
            self.input_shape = (1, self.input_shape[0])
        elif len(self.input_shape) == 2:
            self.input_shape = (self.input_shape[0], self.input_shape[1])
        else:
            raise ValueError("Please check input dimensions.")

        if len(self.output_shape) == 1:
            self.output_shape = (1, self.output_shape[0])
        elif len(self.output_shape) == 2:
            self.output_shape = (self.output_shape[0], self.output_shape[1])
        else:
            raise ValueError("Please check output dimensions.")

    @staticmethod
    def get_activation(function: str):
        blank = ca.MX.sym("blank")

        if function == "sigmoid":
            return ca.Function(function, [blank], [1 / (1 + ca.exp(-blank))])

        if function == "tanh":
            return ca.Function(function, [blank], [ca.tanh(blank)])

        elif function == "relu":
            return ca.Function(function, [blank], [ca.fmax(0, blank)])

        elif function == "softplus":
            return ca.Function(function, [blank], [ca.log(1 + ca.exp(blank))])

        elif function == "gaussian":
            return ca.Function(function, [blank], [ca.exp(-(blank**2))])

        elif function == "linear":
            return ca.Function(function, [blank], [blank])

        else:
            ValueError(f"Unknown activation function:{function}")


class Dense(Layer):
    """
    Fully connected layer.
    """

    def __init__(self, layer: layers.Dense):
        super().__init__(layer)

        # weights and biases
        self.weights, self.biases = layer.get_weights()
        self.biases = self.biases.reshape(1, self.biases.shape[0])

        # check input dimension
        if self.input_shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Please check the input dimensions of this layer. Layer with error: {self.name}"
            )

    def forward(self, input):
        # return forward pass
        return self.activation(input @ self.weights + self.biases)


class Flatten(Layer):
    def forward(self, input):
        # flattens the input
        f = input[0, :]
        for row in range(1, input.shape[0]):
            f = ca.horzcat(f, input[row, :])

        return f


class BatchNormalization(Layer):
    """
    Batch Normalizing layer. Make sure the axis setting is set to two.
    """

    def __init__(self, layer: layers.BatchNormalization):
        super(BatchNormalization, self).__init__(layer)

        # weights and biases
        self.gamma = ca.np.vstack([layer.get_weights()[0]] * self.input_shape[0])
        self.beta = ca.np.vstack([layer.get_weights()[1]] * self.input_shape[0])
        self.mean = ca.np.vstack([layer.get_weights()[2]] * self.input_shape[0])
        self.var = ca.np.vstack([layer.get_weights()[3]] * self.input_shape[0])
        self.epsilon = layer.get_config()["epsilon"]

        # check Dimensions
        if self.input_shape != self.gamma.shape:
            axis = self.config["axis"][0]
            raise ValueError(f"Dimension mismatch. Normalized axis: {axis}")

        # symbolic input layer
        self.input_layer = ca.MX.sym(
            "input_layer", self.input_shape[0], self.input_shape[1]
        )

    def forward(self, input):
        # forward pass
        f = (input - self.mean) / (
            ca.sqrt(self.var + self.epsilon)
        ) * self.gamma + self.beta

        return f


class LSTM(Layer):
    """
    Long Short Term Memory cell.
    """

    def __init__(self, layer: layers.LSTM):
        super(LSTM, self).__init__(layer)

        # recurrent activation
        self.recurrent_activation = self.get_activation(
            layer.get_config()["recurrent_activation"]
        )

        # load weights and biases
        W = layer.get_weights()[0]
        U = layer.get_weights()[1]
        b = layer.get_weights()[2]

        # weights (kernel)
        self.W_i = W[:, : self.units]
        self.W_f = W[:, self.units : self.units * 2]
        self.W_c = W[:, self.units * 2 : self.units * 3]
        self.W_o = W[:, self.units * 3 :]

        # weights (recurrent kernel)
        self.U_i = U[:, : self.units]
        self.U_f = U[:, self.units : self.units * 2]
        self.U_c = U[:, self.units * 2 : self.units * 3]
        self.U_o = U[:, self.units * 3 :]

        # biases
        self.b_i = ca.np.expand_dims(b[: self.units], axis=0)
        self.b_f = ca.np.expand_dims(b[self.units : self.units * 2], axis=0)
        self.b_c = ca.np.expand_dims(b[self.units * 2 : self.units * 3], axis=0)
        self.b_o = ca.np.expand_dims(b[self.units * 3 :], axis=0)

        # initial memory and output
        self.h_0 = ca.np.zeros((1, self.units))
        self.c_0 = ca.np.zeros((1, self.units))

    def forward(self, input):
        # check input shape
        if input.shape != self.input_shape:
            raise ValueError("Dimension mismatch!")

        # initial
        c = self.c_0
        h = self.h_0

        # number of time steps
        steps = self.input_shape[0]

        # forward pass
        for i in range(steps):
            # input for the current step
            x = input[i, :]

            # calculate memory(c) and output(h)
            c, h = self.step(x, c, h)

        # here the output has to be transposed, because of the dense layer implementation
        return h

    def step(self, x_t, c_prev, h_prev):
        # gates
        i_t = self.recurrent_activation(x_t @ self.W_i + h_prev @ self.U_i + self.b_i)
        f_t = self.recurrent_activation(x_t @ self.W_f + h_prev @ self.U_f + self.b_f)
        o_t = self.recurrent_activation(x_t @ self.W_o + h_prev @ self.U_o + self.b_o)
        c_t = self.activation(x_t @ self.W_c + h_prev @ self.U_c + self.b_c)

        # memory and output
        c_next = f_t * c_prev + i_t * c_t
        h_next = o_t * self.activation(c_next)

        return c_next, h_next


class CasadiANN(CasadiPredictor):
    """
    Generic implementations of sequential Keras models in CasADi.
    """

    def __init__(self, serialized_model: SerializedANN):
        """
        Supported layers:
            - Dense (Fully connected layer)
            - Flatten (Reduces the input dimension to 1)
            - BatchNormalizing (Normalization)
            - LSTM (Recurrent Cell)
            - Rescaling
        Args:
            serialized_model: SerializedANN Model.
        """
        super().__init__(serialized_model)

    @property
    def input_shape(self) -> tuple[int, int]:
        """Input shape of Predictor."""
        return 1, self.predictor_model.layers[0].input_shape[1]

    def _build_prediction_function(self) -> ca.Function:
        """Build the prediction function with casadi and a symbolic input."""
        keras_layers = [layer for layer in self.predictor_model.layers]
        casadi_layers = []
        for keras_layer in keras_layers:
            name = keras_layer.get_config()["name"]
            for layer_type in ANNLayerTypes:
                if layer_type.value in name:
                    casadi_layers.append(ann_layer_types[layer_type.value](keras_layer))
                    continue
        function = self.sym_input
        for casadi_layer in casadi_layers:
            function = casadi_layer.forward(function)
        return ca.Function("forward", [self.sym_input], [function])


ann_layer_types = {
    ANNLayerTypes.DENSE: Dense,
    ANNLayerTypes.FLATTEN: Flatten,
    ANNLayerTypes.BATCHNORMALIZATION: BatchNormalization,
    ANNLayerTypes.LSTM: LSTM,
}

casadi_predictors = {
    MLModels.ANN: CasadiANN,
    MLModels.GPR: CasadiGPR,
    MLModels.LINREG: CasadiLinReg,
}
