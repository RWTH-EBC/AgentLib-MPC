import abc
from abc import abstractmethod
import casadi as ca
import numpy as np

from enum import Enum
from keras import layers
from keras.src import Functional
from keras import Sequential
from typing import Union, TYPE_CHECKING

from agentlib_mpc.models.serialized_ml_model import (
    SerializedMLModel,
    SerializedLinReg,
    SerializedGPR,
    SerializedANN,
    MLModels, SerializedKerasANN,
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
    RESCALING = "rescaling"
    RBF = 'rbf'


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
        try:
            self.weights, self.biases = layer.get_weights()
        except ValueError as e:
            if e.__str__() == "not enough values to unpack (expected 2, got 1)":
                self.weights = layer.get_weights()
                self.biases = np.zeros(1)
            else:
                raise e
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
        init = 0
        for inp in input:
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


class Rescaling(Layer):

    def __init__(self, layer: layers.Rescaling):
        super(Rescaling, self).__init__(layer)
        self.offset = layer.offset
        self.scale = layer.scale

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


class FunctionalWrapper:

    def __init__(self, functional: Functional):
        self.functional = CasadiANN.build_prediction_function_functionalAPI(functional)

    def forward(self, input):
        return self.functional(input)
    

class SequentialWrapper:

    def __init__(self, sequential: Sequential):
        self.functional = CasadiANN.build_prediction_function_sequential(sequential)

    def forward(self, input):
        return self.functional(input)


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
        outputs = predictor_model.output_names
        assert len(outputs) == 1, f"Error: Current version only supports Keras Models with one output"
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

        for output in outputs:
            recursive_search(output, len(fnodes[output]) - 1)

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

        _input = [fmx[inp.name, 0] for inp in predictor_model.inputs]
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
}

casadi_predictors = {
    MLModels.ANN: CasadiANN,
    MLModels.GPR: CasadiGPR,
    MLModels.LINREG: CasadiLinReg,
    MLModels.KerasANN: CasadiANN,
}
