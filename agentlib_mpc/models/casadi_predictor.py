import abc
import itertools
from abc import abstractmethod
import casadi as ca
import numpy as np
from agentlib.core.errors import ConfigurationError

from enum import Enum
from keras import layers
from keras.src import Functional
from keras import Sequential
from typing import Iterator, Optional, Union, TYPE_CHECKING

from agentlib_mpc.models.serialized_ml_model import (
    SerializedMLModel,
    SerializedLinReg,
    SerializedGPR,
    SerializedANN,
    MLModels, SerializedKerasANN, SerializedKerasRNN,
)

if TYPE_CHECKING:
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
        self.predictor_model: Union[Sequential, CustomGPR, LinearRegression, Functional] = (
            serialized_model.deserialize()
        )
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
    NORMALIZATION = "normalization"
    CROPPING1D = "cropping1d"
    CONCATENATE = "concatenate"
    RESHAPE = "reshape"
    INPUTSLICE = "input_slice"
    CONSTANT = "constant"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    TRUEDIVIDE = "divide"
    POWER = "power"
    AVERAGE = "average"
    RESCALING = "rescaling"
    RBF = 'rbf'
    LSTM = "lstm"
    GRU = "gru"
    SIMPLERNN = "simple_rnn"


class Layer(abc.ABC):
    """
    Single layer of an artificial neural network.
    """

    def __init__(self, layer: layers.Layer):
        self.config = layer.get_config()

        # name
        if "name" in self.config:
            self.name = self.config["name"]

        # input / output shape
        # TODO: Check if more detailed translation is needed
        if isinstance(layer.input, list):
            self.input_shape = None
        else:
            self.input_shape = layer.input.shape[1:]

        # update the dimensions to two dimensions
        self.update_dimensions()


    def update_dimensions(self):
        """
        CasADi does only work with two dimensional arrays. So the dimensions must be updated.
        """

        if self.input_shape is None:
            pass
        elif len(self.input_shape) == 1:
            self.input_shape = (1, self.input_shape[0])
        elif len(self.input_shape) == 2:
            self.input_shape = (self.input_shape[0], self.input_shape[1])
        else:
            raise ValueError("Please check input dimensions.")

    @staticmethod
    def get_activation(function: str) -> ca.Function:
        blank = ca.MX.sym("blank")

        if function == "sigmoid":
            return ca.Function(function, [blank], [1 / (1 + ca.exp(-blank))])

        elif function == "tanh":
            return ca.Function(function, [blank], [ca.tanh(blank)])

        elif function == "relu":
            return ca.Function(function, [blank], [ca.fmax(0, blank)])

        elif function == 'exponential':
            return ca.Function(function, [blank], [ca.exp(blank)])

        elif function == "softplus":
            return ca.Function(function, [blank], [ca.log(1 + ca.exp(blank))])

        elif function == "gaussian":
            return ca.Function(function, [blank], [ca.exp(-(blank**2))])

        elif function == "linear":
            return ca.Function(function, [blank], [blank])

        elif isinstance(function, dict):
            if 'class_name' in function:
                if 'registered_name' in function:
                    if function['registered_name'] == 'custom_activation>ConcaveActivation':
                        return ca.Function(function['class_name'], [blank],
                                        [-Layer.get_activation(function['config']['activation'])(-blank)])
                    elif function['registered_name'] == 'custom_activation>SaturatedActivation':
                        if function['config']['activation'] == 'relu':
                            return ca.Function(function['class_name'], [blank], [ca.fmin(1, ca.fmax(-1, blank))])
                        elif function['config']['activation'] == 'softplus':
                            casadi_function = ca.if_else(
                                blank >= 0,
                                ca.log((1 + ca.exp(1)) / (1 + ca.exp(1 - blank))),
                                ca.log((1 + ca.exp(1 + blank)) / (1 + ca.exp(1)))
                            )
                            return ca.Function(function['class_name'], [blank], [casadi_function])
                        else:
                            raise NotImplementedError('Keras Model: Saturated activation function for activions other '
                                                      'than relu or softplus are not implemented yet.')

        raise ValueError(f"Unknown activation function:{function}")

    @abstractmethod
    def forward(self, input):
        pass


class Dense(Layer):
    """
    Fully connected layer.
    """

    def __init__(self, layer: layers.Dense):
        super().__init__(layer)

        self.activation = self.get_activation(layer.get_config()["activation"])

        # weights and biases
        weights_list = layer.get_weights()
        self.weights = weights_list[0]
        if len(weights_list) >= 2:
            self.biases = weights_list[1]
        else:
            # create zero bias matching the number of output neurons
            self.biases = np.zeros(self.weights.shape[1])
        self.biases = self.biases.reshape(1, self.biases.shape[0])

        # check input dimension
        # TODO: Check if needed
        if self.input_shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Please check the input dimensions of this layer. Layer with error: {self.name}"
            )

    def forward(self, input):
        # return forward pass
        # TODO: Check if np.repeat is needed
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


    def forward(self, input):
        # forward pass
        f = (input - self.mean) / (
            ca.sqrt(self.var + self.epsilon)
        ) * self.gamma + self.beta

        return f


class Normalization(Layer):

    def __init__(self, layer: layers.Normalization):
        super(Normalization, self).__init__(layer)
        if len(layer.mean.numpy().shape) == 3:
            self.mean = layer.mean.numpy()[-1]
            self.var = layer.variance.numpy()[-1]
        elif len(layer.mean.numpy().shape) == 2:
            self.mean = layer.mean.numpy()
            self.var = layer.variance.numpy()
        else:
            raise Exception(
                f'Normalization layer: Expecting dimension to be 2 or 3, was {len(layer.mean.numpy().shape)}')

    def forward(self, input):
        return (input - np.repeat(self.mean, input.shape[0], axis=0)) / \
            np.repeat(np.sqrt(self.var), input.shape[0], axis=0)


class Cropping1D(Layer):

    def __init__(self, layer: layers.Cropping1D):
        super(Cropping1D, self).__init__(layer)
        self.cropping = layer.cropping

    def forward(self, input):
        return input[self.cropping[0]:input.shape[0] - self.cropping[1], :]


class Concatenate(Layer):

    def __init__(self, layer: layers.Concatenate):
        super(Concatenate, self).__init__(layer)
        self.axis = layer.axis

    def forward(self, *input):
        if self.axis == -1 or self.axis == 2:
            return ca.horzcat(*input)
        elif self.axis == 1:
            return ca.vertcat(*input)
        else:
            raise NotImplementedError(f'Concatenate layer with axis={self.axis} not implemented yet.')


class Reshape(Layer):

    def __init__(self, layer: layers.Reshape):
        super(Reshape, self).__init__(layer)
        self.shape = layer.target_shape

    def forward(self, input):
        return ca.reshape(input, self.shape[0], self.shape[1])


class Add(Layer):
    def __init__(self, layer: layers.Add):
        super(Add, self).__init__(layer)

    def forward(self, *input):
        init = input[0]
        for inp in input[1:]:
            init += inp
        return init


class Subtract(Layer):
    def __init__(self, layer: layers.Subtract):
        super(Subtract, self).__init__(layer)

    def forward(self, *input):
        return input[0] - input[1]
    

class Multiply(Layer):
    def __init__(self, layer: layers.Multiply):
        super(Multiply, self).__init__(layer)

    def forward(self, *input):
        init = input[0]
        for inp in input[1:]:
            init *= inp
        return init
    

class TrueDivide(Layer):
    def __init__(self, layer):
        super(TrueDivide, self).__init__(layer)

    def forward(self, *input):
        return input[0] / input[1]


class Power(Layer):
    def __init__(self, layer):
        super(Power, self).__init__(layer)

    def forward(self, *input):
        return input[0] ** input[1]
    

class Average(Layer):
    def __init__(self, layer):
        super(Average, self).__init__(layer)

    def forward(self, *input):
        init = input[0]
        for inp in input[1:]:
            init += inp
        return init / len(input)


class Rescaling(Layer):

    def __init__(self, layer: layers.Rescaling):
        super(Rescaling, self).__init__(layer)
        # scale and offset can be keras tensors, e.g. when they are derived from the
        # training data, so they are converted to numpy for the use with CasADi
        self.offset = np.asarray(layer.offset, dtype=float)
        self.scale = np.asarray(layer.scale, dtype=float)

    def forward(self, input):
        f = input * self.scale + self.offset
        return f
    
    
class InputSliceLayer(Layer):

    def __init__(self, layer):
        super().__init__(layer)
        self.feature_indices = layer.feature_indices

    def forward(self, input):
        return input[:, self.feature_indices]
    

class ConstantLayer(Layer):

    def __init__(self, layer):
        super().__init__(layer)
        self.constant = ca.DM(layer.constant.numpy())

    def forward(self, input):
        return self.constant


class RBF(Layer):

    def __init__(self, layer):
        super().__init__(layer)
        self.centers = ca.DM(layer.centers.numpy())
        self.log_gamma = ca.DM(layer.log_gamma.numpy())
        self.gamma = ca.exp(self.log_gamma)
        self.units = layer.units

    def forward(self, input):
        input_repm = ca.repmat(input, self.units, 1)
        diff = input_repm - self.centers
        distance_sq = ca.sum2(diff**2)
        phi = ca.exp(-self.gamma * distance_sq)
        return phi.T


class RecurrentLayer(Layer, abc.ABC):
    """
    Base class for recurrent layers.

    Recurrent layers are translated cell-wise, i.e. the forward pass unrolls the input
    sequence and applies ``step`` once per row of the input. The same implementation
    can therefore be used for a full sequence (as during training) and for a single
    time step, which is what multiple shooting in the MPC needs.

    The forward pass always returns a tuple ``(output, *states)``.
    """

    def __init__(self, layer: layers.Layer):
        super().__init__(layer)
        self.units: int = self.config["units"]
        self.activation = self.get_activation(self.config["activation"])
        self.return_sequences: bool = self.config.get("return_sequences", False)
        for unsupported in ("go_backwards", "stateful", "unroll"):
            if self.config.get(unsupported):
                raise NotImplementedError(
                    f'Recurrent layer "{self.name}" was configured with '
                    f"'{unsupported}=True', which is not supported in CasADi."
                )

    @property
    @abstractmethod
    def number_of_states(self) -> int:
        """Number of state tensors the layer carries between two time steps."""

    @abstractmethod
    def step(self, x_t, *states):
        """Performs a single time step and returns the new states. Following keras,
        the first state is the layer output (h) by convention."""

    def forward(self, x, *states):
        if not states:
            states = tuple(
                np.zeros((1, self.units)) for _ in range(self.number_of_states)
            )
        if len(states) != self.number_of_states:
            raise ValueError(
                f'Recurrent layer "{self.name}" expects {self.number_of_states} '
                f"initial states, but got {len(states)}."
            )

        outputs = []
        for n in range(x.shape[0]):
            states = self.step(x[n, :], *states)
            outputs.append(states[0])

        if self.return_sequences:
            output = ca.vertcat(*outputs)
        else:
            output = states[0]
        return (output, *states)


class SimpleRNN(RecurrentLayer):
    """Fully connected recurrent unit."""

    def __init__(self, layer: layers.SimpleRNN):
        super().__init__(layer)
        weights = layer.get_weights()
        self.W = weights[0]
        self.W_rec = weights[1]
        if len(weights) >= 3:
            self.b = weights[2].reshape(1, -1)
        else:
            self.b = np.zeros((1, self.units))

    @property
    def number_of_states(self) -> int:
        return 1

    def step(self, x_t, h_prev):
        h = self.activation(x_t @ self.W + self.b + h_prev @ self.W_rec)
        return (h,)


class LSTM(RecurrentLayer):
    """Long short term memory cell."""

    def __init__(self, layer: layers.LSTM):
        super().__init__(layer)
        self.recurrent_activation = self.get_activation(
            self.config["recurrent_activation"]
        )

        weights = layer.get_weights()
        kernel = weights[0]
        recurrent_kernel = weights[1]
        if len(weights) >= 3:
            bias = weights[2]
        else:
            bias = np.zeros(self.units * 4)

        u = self.units
        # keras stores the gates in the order input, forget, cell, output
        self.W_i, self.W_f, self.W_c, self.W_o = (
            kernel[:, :u],
            kernel[:, u : u * 2],
            kernel[:, u * 2 : u * 3],
            kernel[:, u * 3 :],
        )
        self.U_i, self.U_f, self.U_c, self.U_o = (
            recurrent_kernel[:, :u],
            recurrent_kernel[:, u : u * 2],
            recurrent_kernel[:, u * 2 : u * 3],
            recurrent_kernel[:, u * 3 :],
        )
        self.b_i, self.b_f, self.b_c, self.b_o = (
            bias[:u].reshape(1, -1),
            bias[u : u * 2].reshape(1, -1),
            bias[u * 2 : u * 3].reshape(1, -1),
            bias[u * 3 :].reshape(1, -1),
        )

    @property
    def number_of_states(self) -> int:
        return 2

    def step(self, x_t, h_prev, c_prev):
        i_t = self.recurrent_activation(x_t @ self.W_i + h_prev @ self.U_i + self.b_i)
        f_t = self.recurrent_activation(x_t @ self.W_f + h_prev @ self.U_f + self.b_f)
        o_t = self.recurrent_activation(x_t @ self.W_o + h_prev @ self.U_o + self.b_o)
        c_hat = self.activation(x_t @ self.W_c + h_prev @ self.U_c + self.b_c)

        c_next = f_t * c_prev + i_t * c_hat
        h_next = o_t * self.activation(c_next)
        return h_next, c_next


class GRU(RecurrentLayer):
    """Gated recurrent unit."""

    def __init__(self, layer: layers.GRU):
        super().__init__(layer)
        self.recurrent_activation = self.get_activation(
            self.config["recurrent_activation"]
        )
        self.reset_after: bool = self.config.get("reset_after", True)

        weights = layer.get_weights()
        kernel = weights[0]
        recurrent_kernel = weights[1]

        u = self.units
        # keras stores the gates in the order update (z), reset (r), candidate (h)
        self.W_z, self.W_r, self.W_h = (
            kernel[:, :u],
            kernel[:, u : u * 2],
            kernel[:, u * 2 :],
        )
        self.U_z, self.U_r, self.U_h = (
            recurrent_kernel[:, :u],
            recurrent_kernel[:, u : u * 2],
            recurrent_kernel[:, u * 2 :],
        )

        if len(weights) >= 3:
            bias = weights[2]
        elif self.reset_after:
            bias = np.zeros((2, u * 3))
        else:
            bias = np.zeros(u * 3)

        if self.reset_after:
            # with reset_after, keras keeps separate input and recurrent biases
            input_bias, recurrent_bias = bias[0], bias[1]
            self.br_z, self.br_r, self.br_h = (
                recurrent_bias[:u].reshape(1, -1),
                recurrent_bias[u : u * 2].reshape(1, -1),
                recurrent_bias[u * 2 :].reshape(1, -1),
            )
        else:
            input_bias = bias
            self.br_z = self.br_r = self.br_h = np.zeros((1, u))
        self.b_z, self.b_r, self.b_h = (
            input_bias[:u].reshape(1, -1),
            input_bias[u : u * 2].reshape(1, -1),
            input_bias[u * 2 :].reshape(1, -1),
        )

    @property
    def number_of_states(self) -> int:
        return 1

    def step(self, x_t, h_prev):
        z = self.recurrent_activation(
            x_t @ self.W_z + self.b_z + h_prev @ self.U_z + self.br_z
        )
        r = self.recurrent_activation(
            x_t @ self.W_r + self.b_r + h_prev @ self.U_r + self.br_r
        )

        if self.reset_after:
            # the reset gate is applied after the matrix multiplication
            recurrent_h = r * (h_prev @ self.U_h + self.br_h)
        else:
            recurrent_h = (r * h_prev) @ self.U_h
        h_hat = self.activation(x_t @ self.W_h + self.b_h + recurrent_h)

        h = z * h_prev + (1 - z) * h_hat
        return (h,)


class _ModelWrapper:
    """Base class for nested keras models which are used as a layer."""

    functional: ca.Function

    def forward(self, *input):
        result = self.functional(*input)
        # a casadi function with a single output returns the MX directly, whereas
        # multiple outputs are returned as a list. Nested models with multiple outputs
        # are accessed through their tensor_index, so they have to stay a tuple.
        if isinstance(result, (list, tuple)):
            return tuple(result)
        return result


class FunctionalWrapper(_ModelWrapper):

    def __init__(self, functional: Functional):
        self.functional = CasadiANN.build_prediction_function_functionalAPI(functional)


class SequentialWrapper(_ModelWrapper):

    def __init__(self, sequential: Sequential):
        self.functional = CasadiANN.build_prediction_function_sequential(sequential)


class CasadiANN(CasadiPredictor):
    """
    Generic implementations of sequential Keras models in CasADi.
    """

    def __init__(self, serialized_model: Union[SerializedANN, SerializedKerasANN]):
        """
        Supported layers:
            - Dense (Fully connected layer)
            - Flatten (Reduces the input dimension to 1)
            - BatchNormalizing
            - Normalizing
            - Cropping1D
            - Concatenate
            - Reshape
            - Add
            - Rescaling
        Args:
            serialized_model: SerializedANN or SerializedKerasANN Model.
        """
        super().__init__(serialized_model)

    @property
    def input_shape(self) -> tuple[int, int]:
        """Input shape of Predictor."""
        assert len(self.predictor_model.input_shape) == 2, (f"Error: Current version only supports Keras Models with "
                                                            f"input_shape length 2, but was "
                                                            f"{len(self.predictor_model.input_shape)}")
        assert isinstance(self.predictor_model.input_shape[1], int), (f"Error: Current version only supports "
                                                                      f"Keras Models with 1 input layer, but was "
                                                                      f"{len(self.predictor_model.input_shape)}")
        return 1, self.predictor_model.input_shape[1]

    def _build_prediction_function(self) -> ca.Function:
        """Build the prediction function with casadi and a symbolic input."""
        if isinstance(self.predictor_model, Functional):
            return self.build_prediction_function_functionalAPI(self.predictor_model)
        elif not isinstance(self.predictor_model, Sequential):
            raise NotImplementedError(f"Error: Keras Model type {type(self.predictor_model)} not supported")
        else:
            return self.build_prediction_function_sequential(self.predictor_model)
    
    @staticmethod
    def build_prediction_function_sequential(predictor_model) -> ca.Function:
        keras_layers = [layer for layer in predictor_model.layers]
        casadi_layers = []
        for keras_layer in keras_layers:
            name = keras_layer.get_config()["name"]
            for layer_type in ANNLayerTypes:
                if layer_type.value in name:
                    casadi_layers.append(ann_layer_types[layer_type](keras_layer))
                    break
            else:
                raise NotImplementedError(f'Keras Layer with type "{name}" is not supported yet.')
        sym_input = ca.MX.sym("input", 1, predictor_model.input_shape[1])
        function = sym_input
        for casadi_layer in casadi_layers:
            function = casadi_layer.forward(function)
        return ca.Function("forward", [sym_input], [function])

    @staticmethod
    def build_prediction_function_functionalAPI(predictor_model) -> ca.Function:

        fmx = {}
        fnodes = {}
        flayers = {}

        # Add Layers
        for layer in predictor_model.layers:

            # get the name of the layer
            name = layer.get_config()['name']

            # recreate the matching layer
            if 'input' in name and 'slice' not in name:
                if len(layer.batch_shape) > 2:
                    if layer.batch_shape[1] is None:
                        fmx[name, 0] = ca.MX.sym('input_layer', 1, layer.batch_shape[2])
                    else:
                        fmx[name, 0] = ca.MX.sym('input_layer', layer.batch_shape[1], layer.batch_shape[2])
                else:
                    fmx[name, 0] = ca.MX.sym('input_layer', 1, layer.batch_shape[1])
            else:
                for layer_type in ANNLayerTypes:
                    if layer_type.value in name:
                        ca_layer = ann_layer_types[layer_type](layer)
                        flayers[name] = ca_layer
                        break
                else:
                    if isinstance(layer, Functional):
                        flayers[name] = FunctionalWrapper(layer)
                    elif isinstance(layer, Sequential):
                        flayers[name] = SequentialWrapper(layer)
                    else:
                        raise NotImplementedError(f'Keras Layer with type "{name}" is not supported yet.')

        # Create Nodes
        for layer in predictor_model.layers:
            connections = []
            for node in layer._inbound_nodes:
                connection = []
                for it in node.input_tensors:
                    keras_history = it._keras_history
                    inbound_layer = keras_history.operation
                    node_index = keras_history.node_index
                    tensor_index = keras_history.tensor_index
                    connection.append([inbound_layer.name, node_index, tensor_index])
                connections.append(connection)
            fnodes[layer.get_config()['name']] = connections

        # Order Nodes
        # the keras history is used instead of 'output_names', because models with
        # multiple outputs (e.g. a recurrent model returning its states) can have the
        # same layer name multiple times in 'output_names'
        outputs = [
            (it._keras_history.operation.name, it._keras_history.node_index)
            for it in predictor_model.outputs
        ]
        ordering = []
        visited_notes = []

        def recursive_search(name, depth):
            input_nodes = fnodes[name][depth]
            if len(input_nodes) > 0:
                for input_node in input_nodes:
                    if (input_node[0], input_node[1]) not in visited_notes:
                        recursive_search(input_node[0], input_node[1])
            visited_notes.append((name, depth))
            ordering.append((name, depth))

        for output_name, output_depth in outputs:
            if (output_name, output_depth) not in visited_notes:
                recursive_search(output_name, output_depth)

        # Update Forward
        for name, depth in ordering:
            if 'input' in name and 'slice' not in name:
                continue
            else:
                input_nodes = fnodes[name][depth]
                input = []
                if len(input_nodes) > 1:
                    for input_node in input_nodes:
                        i = fmx[input_node[0], input_node[1]]
                        if isinstance(i, tuple):
                            input.append(i[input_node[2]])
                        else:
                            input.append(i)
                    output = flayers[name].forward(*input)
                else:
                    input = fmx[input_nodes[0][0], input_nodes[0][1]]
                    if isinstance(input, tuple):
                        input = input[input_nodes[0][2]]
                    output = flayers[name].forward(input)
                fmx[name, depth] = output

        _input = [
            fmx[inp._keras_history.operation.name, 0]
            for inp in predictor_model.inputs
        ]
        prediction = []
        for it in predictor_model.outputs:
            keras_history = it._keras_history
            inbound_layer = keras_history.operation
            node_index = keras_history.node_index
            tensor_index = keras_history.tensor_index
            mx_var = fmx[inbound_layer.name, node_index]
            if isinstance(mx_var, tuple):
                mx_var = mx_var[tensor_index]
            prediction.append(mx_var)

        return ca.Function("forward", _input, prediction)


class CasadiRNN(CasadiPredictor):
    """
    Translates a trained multi step keras model (SimpleRNN / LSTM / GRU) into a
    single step CasADi function.

   	Only the recurrent cell of the model is translated here, and it is
    translated for a sequence length of one. The hidden states of the recurrent layer
    become additional in- and outputs of the prediction function, so the optimization
    backend can treat them like any other state and link them over the horizon with
    multiple shooting constraints.

    The translated part of the keras model is the part which performs a single
    recurrent step, i.e. the model that maps ``[sequence, *states]`` to
    ``[prediction, *states]`` for a sequence of arbitrary length. Multi step models
    are usually saved with that part as a nested model, next to a part that derives
    the initial states from a warmup sequence. The latter is not used here, since the
    optimization backend initializes the states with zeros and rolls them out over
    measured past data instead.

    The step model is identified by its signature. If a model contains several models
    with that signature, the one to use has to be named in the serialized model.

    Attributes:
        step_model: The keras model that performs a single recurrent step.
        state_dimensions: Number of units of each state tensor the model carries over.
    """

    def __init__(self, serialized_model: SerializedKerasRNN) -> None:
        self.serialized_model: SerializedKerasRNN = serialized_model
        self.predictor_model: Union[Sequential, Functional] = (
            serialized_model.deserialize()
        )
        self.step_model: Functional = self._find_step_model(
            self.predictor_model, name=serialized_model.step_model
        )
        (
            self._sequence_input,
            self._state_inputs,
            self._sequence_output,
            self._state_outputs,
        ) = self._sort_step_model_signature(self.step_model)
        self.state_dimensions: list[int] = [
            int(self.step_model.inputs[i].shape[-1]) for i in self._state_inputs
        ]
        self.sym_input: ca.MX = self._get_sym_input()
        self.prediction_function: ca.Function = self._build_prediction_function()

    @property
    def input_shape(self) -> tuple[int, int]:
        """Input shape of one time step of the predictor."""
        return 1, int(self.step_model.inputs[self._sequence_input].shape[-1])

    @property
    def output_shape(self) -> tuple[int, int]:
        """Output shape of one time step of the predictor."""
        return 1, int(self.step_model.outputs[self._sequence_output].shape[-1])

    @property
    def state_dimension(self) -> int:
        """Total number of scalar hidden states carried between two time steps."""
        return sum(self.state_dimensions)

    def _get_sym_input(self):
        """Returns the symbolic input of a single time step as a column vector."""
        return ca.MX.sym("input", self.input_shape[1], 1)

    @staticmethod
    def _is_step_model(model) -> bool:
        """Checks whether a keras model maps [sequence, *states] to
        [prediction, *states] for a sequence of arbitrary length."""
        if not isinstance(model, Functional):
            return False
        # next to the sequence, a step model takes and returns the hidden states
        if len(model.inputs) < 2 or len(model.outputs) < 2:
            return False
        sequences = [inp for inp in model.inputs if len(inp.shape) == 3]
        if len(sequences) != 1:
            return False
        # the length of the sequence has to be free, so it can be applied to a single
        # time step. A model initializing the states takes a warmup sequence of fixed
        # length instead.
        return sequences[0].shape[1] is None

    @classmethod
    def _nested_models(cls, model) -> Iterator[Functional]:
        """Yields all keras models nested inside the given model."""
        for layer in getattr(model, "layers", ()):
            if isinstance(layer, Functional):
                yield layer
                yield from cls._nested_models(layer)

    @classmethod
    def _find_step_model(cls, model, name: Optional[str] = None) -> Functional:
        """Finds the keras model that performs a single recurrent step.

        Args:
            model: The saved multi step model.
            name: Name of the step model. If None, it is identified by its signature.
        """
        if name is not None:
            for candidate in itertools.chain([model], cls._nested_models(model)):
                if candidate.name == name:
                    cls._assert_step_model(candidate)
                    return candidate
            available = [nested.name for nested in cls._nested_models(model)]
            raise ConfigurationError(
                f"The serialized model declares '{name}' as the model performing a "
                f"single recurrent step, but '{model.name}' does not contain a model "
                f"of that name. Available: {available}."
            )

        if cls._is_step_model(model):
            return model
        candidates = [
            nested for nested in cls._nested_models(model) if cls._is_step_model(nested)
        ]
        if len(candidates) == 1:
            return candidates[0]

        if not candidates:
            raise ConfigurationError(
                f'The keras model "{model.name}" does not contain a model which maps '
                f"[sequence, *states] to [prediction, *states] for a sequence of "
                f"arbitrary length, so it cannot be evaluated one step at a time. "
                f"Please check that this is a multi step model."
            )
        raise ConfigurationError(
            f'The keras model "{model.name}" contains more than one model which '
            f"performs a single recurrent step: {[c.name for c in candidates]}. "
            f"Please declare which one to use with 'step_model' in the serialized "
            f"model."
        )

    @classmethod
    def _assert_step_model(cls, model):
        """Raises an error if the model does not perform a single recurrent step."""
        if cls._is_step_model(model):
            return
        raise ConfigurationError(
            f'The model "{model.name}" does not have the expected signature '
            f"[sequence, *states] -> [prediction, *states] for a sequence of "
            f"arbitrary length. Inputs: {[inp.shape for inp in model.inputs]}, "
            f"outputs: {[out.shape for out in model.outputs]}."
        )

    @staticmethod
    def _sort_step_model_signature(
        step_model: Functional,
    ) -> tuple[int, list[int], int, list[int]]:
        """Sorts the in- and outputs of the step model into the sequence and the
        hidden states, and returns their indices."""

        def split(tensors, kind: str) -> tuple[int, list[int]]:
            sequences = [i for i, t in enumerate(tensors) if len(t.shape) == 3]
            states = [i for i, t in enumerate(tensors) if len(t.shape) == 2]
            if len(sequences) != 1 or len(sequences) + len(states) != len(tensors):
                raise NotImplementedError(
                    f"Expected exactly one sequence and any number of states as the "
                    f'{kind} of the recurrent model "{step_model.name}", but got '
                    f"shapes {[t.shape for t in tensors]}."
                )
            return sequences[0], states

        sequence_input, state_inputs = split(step_model.inputs, "inputs")
        sequence_output, state_outputs = split(step_model.outputs, "outputs")

        in_dims = [step_model.inputs[i].shape[-1] for i in state_inputs]
        out_dims = [step_model.outputs[i].shape[-1] for i in state_outputs]
        if in_dims != out_dims:
            raise NotImplementedError(
                f'The states of the recurrent model "{step_model.name}" do not match '
                f"between in- and output. Got {in_dims} and {out_dims}."
            )
        return sequence_input, state_inputs, sequence_output, state_outputs

    def _build_prediction_function(self) -> ca.Function:
        """Builds a CasADi function performing one time step of the recurrent model.

        The resulting function maps the input features and the hidden states at time k
        to the prediction at time k+1 and the hidden states at time k+1. All vectors
        are column vectors.
        """
        keras_function = CasadiANN.build_prediction_function_functionalAPI(
            self.step_model
        )

        sym_states = [
            ca.MX.sym(f"state_{i}", dim, 1)
            for i, dim in enumerate(self.state_dimensions)
        ]
        # the layer implementations work on rows, the models in agentlib_mpc work on
        # column vectors, so the inputs are transposed here and the outputs back
        arguments: list[Union[ca.MX, None]] = [None] * len(self.step_model.inputs)
        arguments[self._sequence_input] = self.sym_input.T
        for slot, state in zip(self._state_inputs, sym_states):
            arguments[slot] = state.T

        results = keras_function(*arguments)
        if not isinstance(results, (list, tuple)):
            results = [results]

        prediction = results[self._sequence_output].T
        next_states = [results[slot].T for slot in self._state_outputs]

        return ca.Function(
            "rnn_step",
            [self.sym_input, *sym_states],
            [prediction, *next_states],
            ["input", *[f"state_{i}" for i in range(len(sym_states))]],
            ["output", *[f"next_state_{i}" for i in range(len(sym_states))]],
        )

    def predict(
        self, x: Union[np.ndarray, ca.MX], states: Union[np.ndarray, ca.MX] = None
    ) -> list[Union[ca.DM, ca.MX]]:
        """
        Performs one time step of the recurrent model.

        Args:
            x: input features of the current time step.
            states: all hidden states of the current time step, stacked into a single
                column vector. Defaults to zeros.

        Returns:
            A list holding the prediction for the next time step, followed by the
            hidden states of the next time step stacked into a single column vector.
        """
        if states is None:
            states = ca.DM.zeros(self.state_dimension, 1)
        result = self.prediction_function(x, *self.split_states(states))
        return [result[0], ca.vertcat(*result[1:])]

    def split_states(
        self, states: Union[np.ndarray, ca.MX]
    ) -> list[Union[np.ndarray, ca.MX]]:
        """Splits a stacked vector of hidden states into the state tensors of the
        recurrent layer."""
        split = []
        offset = 0
        for dim in self.state_dimensions:
            split.append(states[offset : offset + dim])
            offset += dim
        return split


ann_layer_types = {
    ANNLayerTypes.DENSE: Dense,
    ANNLayerTypes.FLATTEN: Flatten,
    ANNLayerTypes.BATCHNORMALIZATION: BatchNormalization,
    ANNLayerTypes.NORMALIZATION: Normalization,
    ANNLayerTypes.CROPPING1D: Cropping1D,
    ANNLayerTypes.CONCATENATE: Concatenate,
    ANNLayerTypes.RESHAPE: Reshape,
    ANNLayerTypes.INPUTSLICE: InputSliceLayer,
    ANNLayerTypes.CONSTANT: ConstantLayer,
    ANNLayerTypes.ADD: Add,
    ANNLayerTypes.SUBTRACT: Subtract,
    ANNLayerTypes.MULTIPLY: Multiply,
    ANNLayerTypes.TRUEDIVIDE: TrueDivide,
    ANNLayerTypes.POWER: Power,
    ANNLayerTypes.RESCALING: Rescaling,
    ANNLayerTypes.RBF: RBF,
    ANNLayerTypes.AVERAGE: Average,
    ANNLayerTypes.LSTM: LSTM,
    ANNLayerTypes.GRU: GRU,
    ANNLayerTypes.SIMPLERNN: SimpleRNN,
}

casadi_predictors = {
    MLModels.ANN: CasadiANN,
    MLModels.GPR: CasadiGPR,
    MLModels.LINREG: CasadiLinReg,
    MLModels.KerasANN: CasadiANN,
    MLModels.KerasRNN: CasadiRNN,
}
