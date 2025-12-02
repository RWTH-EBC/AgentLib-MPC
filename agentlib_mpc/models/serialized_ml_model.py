import abc
import json
import logging
import subprocess

import numpy as np

from enum import Enum
from copy import deepcopy
from keras import Sequential
from keras.src import Functional
from pathlib import Path
from pydantic import ConfigDict, Field, BaseModel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF
from sklearn.linear_model import LinearRegression
from typing import Union, Optional

from agentlib_mpc.data_structures.ml_model_datatypes import OutputFeature, Feature

logger = logging.getLogger(__name__)


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


class MLModels(str, Enum):
    ANN = "ANN"
    GPR = "GPR"
    LINREG = "LinReg"
    KerasANN = "KerasANN"


class SerializedMLModel(BaseModel, abc.ABC):
    dt: Union[float, int] = Field(
        title="dt",
        description="The length of time step of one prediction of Model in seconds.",
    )
    input: dict[str, Feature] = Field(
        default=None,
        title="input",
        description="Model input variables with their lag order.",
    )
    output: dict[str, OutputFeature] = Field(
        default=None,
        title="output",
        description="Model output variables (which are automatically also inputs, as "
        "we need them recursively in MPC.) with their lag order.",
    )
    agentlib_mpc_hash: str = Field(
        default_factory=get_git_revision_short_hash,
        description="The commit hash of the agentlib_mpc version this was created with.",
    )
    training_info: Optional[dict] = Field(
        default=None,
        title="Training Info",
        description="Config of Trainer class with all the meta data used for training of the Model.",
    )

    model_type: MLModels
    model_config = ConfigDict(protected_namespaces=())

    @classmethod
    @abc.abstractmethod
    def serialize(
        cls,
        model: Union[Sequential, GaussianProcessRegressor, LinearRegression],
        dt: Union[float, int],
        input: dict[str, Feature],
        output: dict[str, OutputFeature],
        training_info: Optional[dict] = None,
    ):
        """
        Args:
            model:  Machine Learning Model.
            dt:     The length of time step of one prediction of Model in seconds.
            input:  Model input variables with their lag order.
            output: Model output variables (which are automatically also inputs, as
                    we need them recursively in MPC.) with their lag order.
            training_info: Config of Trainer Class, which trained the Model.
        Returns:
            SerializedMLModel version of the passed ML Model.
        """
        pass

    @abc.abstractmethod
    def deserialize(self):
        """
        Deserializes SerializedMLModel object and returns a specific Machine Learning Model object.
        Returns:
            MLModel: Machine Learning Model.
        """
        pass

    def save_serialized_model(self, path: Path):
        """
        Saves MLModel object as json string.
        Args:
            path: relative/absolute path which determines where the json will be saved.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json())

        # with open(path, "w") as json_file:
        #     json_file.write(self.model_dump_json())
        # Displays the file path under which the json file has been saved.
        logger.info(f"Model has been saved under the following path: {path}")

    @classmethod
    def load_serialized_model_from_file(cls, path: Path):
        """
        Loads SerializedMLModel object from a json file and creates a new specific Machine Learning Model object
        which is returned.

        Args:
            path: relative/absolute path which determines which json file will be loaded.
        Returns:
            SerializedMLModel object with data from json file.
        """
        with open(path, "r") as json_file:
            model_data = json.load(json_file)
        return cls.load_serialized_model_from_dict(model_data)

    @classmethod
    def load_serialized_model_from_string(cls, json_string: str):
        """
        Loads SerializedMLModel object from a json string and creates a new specific Machine Learning Model object
        which is returned.

        Args:
            json_string: json string which will be loaded.
        Returns:
            SerializedMLModel object with data from json file.
        """
        model_data = json.loads(json_string)
        return cls.load_serialized_model_from_dict(model_data)

    @classmethod
    def load_serialized_model_from_dict(cls, model_data: dict):
        """
        Loads SerializedMLModel object from a dict and creates a new specific Machine Learning Model object
        which is returned.

        Args:
            json_string: json string which will be loaded.
        Returns:
            SerializedMLModel object with data from json file.
        """
        model_type = model_data["model_type"]
        return serialized_models[model_type](**model_data)

    @classmethod
    def load_serialized_model(cls, model_data: Union[dict, str, Path]):
        """Loads the ML model from a source"""
        if isinstance(model_data, dict):
            return cls.load_serialized_model_from_dict(model_data)
        if isinstance(model_data, (str, Path)):
            if Path(model_data).exists():
                return cls.load_serialized_model_from_file(model_data)
        return cls.load_serialized_model_from_string(model_data)


class SerializedANN(SerializedMLModel):
    """
    Contains Keras ANN in serialized form and offers functions to transform
    Keras Sequential ANNs to SerializedANN objects (from_ANN) and vice versa (deserialize).

    attributes:
        structure: architecture/structure of ANN saved as json string.
        weights: weights and biases of all layers saved as lists of np.ndarrays.
    """

    weights: list[list] = Field(
        default=None,
        title="weights",
        description="The weights of the ANN.",
    )
    structure: str = Field(
        default=None,
        title="structure",
        description="The structure of the ANN as json string.",
    )
    optimizer: str = Field(
        default=None,
        title="optimizer",
        description="The optimizer used for training the ANN.",
    )
    loss: str = Field(
        default=None,
        title="loss",
        description="The loss function used for training the ANN.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_type: MLModels = MLModels.ANN

    @classmethod
    def serialize(
        cls,
        model: Sequential,
        dt: Union[float, int],
        input: dict[str, Feature],
        output: dict[str, OutputFeature],
        training_info: Optional[dict] = None,
    ):
        """Serializes Keras Sequential ANN and returns SerializedANN object"""
        structure = model.to_json()
        weights = []
        for layer in model.layers:
            weight_l = layer.get_weights()
            for idx in range(len(weight_l)):
                weight_l[idx] = weight_l[idx].tolist()
            weights.append(weight_l)

        optimizer = model.optimizer.__class__.__name__.lower()
        loss = model.loss

        return cls(
            structure=structure,
            weights=weights,
            dt=dt,
            input=input,
            output=output,
            trainer_config=training_info,
            optimizer=optimizer,
            loss=loss,
        )

    def deserialize(self) -> Sequential:
        """Deserializes SerializedANN object and returns a Keras Sequential ANN."""
        from keras import models

        ann = models.model_from_json(self.structure)
        layer_weights = []
        for layer in self.weights:
            l_weight = []
            layer_weights.append(l_weight)
            for matrix in layer:
                l_weight.append(np.asarray(matrix))

        for i, layer in enumerate(ann.layers):
            layer.set_weights(layer_weights[i])
        return ann

    def to_dict(self) -> dict:
        """Transforms self to a dictionary and the numpy arrays to lists, so they can
        be serialized."""
        ann_dict = deepcopy(self.__dict__)
        for layer in ann_dict["weights"]:
            for idx in range(0, len(layer)):
                layer[idx] = layer[idx].tolist()
        return ann_dict


class GPRDataHandlingParameters(BaseModel):
    normalize: bool = Field(
        default=False,
        title="normalize",
        description="Boolean which defines whether the input data will be normalized or not.",
    )
    scale: float = Field(
        default=1.0,
        title="scale",
        description="Number by which the y vector is divided before training and multiplied after evaluation.",
    )
    mean: Optional[list] = Field(
        default=None,
        title="mean",
        description="Mean values of input data for normalization. None if normalize equals to False.",
    )
    std: Optional[list] = Field(
        default=None,
        title="standard deviation",
        description="Standard deviation of input data for normalization. None if normalize equals to False.",
    )


class CustomGPR(GaussianProcessRegressor):
    """
    Extends scikit-learn GaussianProcessRegressor with normalizing and scaling option
    by adding the attribute data_handling, customizing the predict function accordingly
    and adding a normalize function.
    """

    def __init__(
        self,
        kernel=None,
        *,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        random_state=None,
        data_handling=None,
    ):
        super().__init__(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state,
        )
        if data_handling is None:
            data_handling = GPRDataHandlingParameters()
        self.data_handling: GPRDataHandlingParameters = data_handling

    def predict(self, X, return_std=False, return_cov=False):
        """
        Overwrite predict method of GaussianProcessRegressor to include normalization.
        """
        if self.data_handling.normalize:
            X = self._normalize(X)
        return super().predict(X, return_std, return_cov)

    def _normalize(self, x: np.ndarray):
        mean = self.data_handling.mean
        std = self.data_handling.std

        if mean is None and std is not None:
            raise ValueError("Mean and std are not valid.")

        return (x - mean) / std


class GPRKernelParameters(BaseModel):
    constant_value: float = Field(
        default=1.0,
        title="constant value",
        description="The constant value which defines the covariance: k(x_1, x_2) = constant_value.",
    )
    constant_value_bounds: Union[tuple, str] = Field(
        default=(1e-5, 1e5),
        title="constant value bounds",
        description="The lower and upper bound on constant_value. If set to “fixed”, "
        "constant_value cannot be changed during hyperparameter tuning.",
    )
    length_scale: Union[float, list] = Field(
        default=1.0,
        title="length_scale",
        description="The length scale of the kernel. If a float, an isotropic kernel "
        "is used. If an array, an anisotropic kernel is used where each "
        "dimension of l defines the length-scale of the respective feature "
        "dimension.",
    )
    length_scale_bounds: Union[tuple, str] = Field(
        default=(1e-5, 1e5),
        title="length_scale_bounds",
        description="The lower and upper bound on ‘length_scale’. If set to “fixed”, "
        "‘length_scale’ cannot be changed during hyperparameter tuning.",
    )
    noise_level: float = Field(
        default=1.0,
        title="noise level",
        description="Parameter controlling the noise level (variance).",
    )
    noise_level_bounds: Union[tuple, str] = Field(
        default=(1e-5, 1e5),
        title="noise level bounds",
        description="The lower and upper bound on ‘noise_level’. If set to “fixed”, "
        "‘noise_level’ cannot be changed during hyperparameter tuning.",
    )
    theta: list = Field(
        title="theta",
        description="Returns the (flattened, log-transformed) non-fixed gpr_parameters.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_model(cls, model: CustomGPR) -> "GPRKernelParameters":
        return cls(
            constant_value=model.kernel_.k1.k1.constant_value,
            constant_value_bounds=model.kernel_.k1.k1.constant_value_bounds,
            length_scale=model.kernel_.k1.k2.length_scale,
            length_scale_bounds=model.kernel_.k1.k2.length_scale_bounds,
            noise_level=model.kernel_.k2.noise_level,
            noise_level_bounds=model.kernel_.k2.noise_level_bounds,
            theta=model.kernel_.theta.tolist(),
        )


class GPRParameters(BaseModel):
    alpha: Union[float, list] = Field(
        default=1e-10,
        title="alpha",
        description="Value added to the diagonal of the kernel matrix during fitting. "
        "This can prevent a potential numerical issue during fitting, by "
        "ensuring that the calculated values form a positive definite matrix. "
        "It can also be interpreted as the variance of additional Gaussian "
        "measurement noise on the training observations. Note that this is "
        "different from using a WhiteKernel. If an array is passed, it must "
        "have the same number of entries as the data used for fitting and is "
        "used as datapoint-dependent noise level. Allowing to specify the "
        "noise level directly as a parameter is mainly for convenience and "
        "for consistency with Ridge.",
    )
    L: list = Field(
        title="L",
        description="Lower-triangular Cholesky decomposition of the kernel in X_train.",
    )
    X_train: list = Field(
        title="X_train",
        description="Feature vectors or other representations of training data (also "
        "required for prediction).",
    )
    y_train: list = Field(
        title="y_train",
        description="Target values in training data (also required for prediction).",
    )
    n_features_in: int = Field(
        title="number of input features",
        description="Number of features seen during fit.",
    )
    log_marginal_likelihood_value: float = Field(
        title="log marginal likelihood value",
        description="The log-marginal-likelihood of self.kernel_.theta.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_model(cls, model: CustomGPR) -> "GPRParameters":
        return cls(
            alpha=model.alpha_.tolist(),
            L=model.L_.tolist(),
            X_train=model.X_train_.tolist(),
            y_train=model.y_train_.tolist(),
            n_features_in=model.n_features_in_,
            log_marginal_likelihood_value=model.log_marginal_likelihood_value_,
        )


class SerializedGPR(SerializedMLModel):
    """
    Contains scikit-learn GaussianProcessRegressor and its Kernel and provides functions to transform
    these to SerializedGPR objects and vice versa.

    Attributes:

    """

    data_handling: GPRDataHandlingParameters = Field(
        default=None,
        title="data_handling",
        description="Information about data handling for GPR.",
    )
    kernel_parameters: GPRKernelParameters = Field(
        default=None,
        title="kernel parameters",
        description="Parameters of kernel of the fitted GPR.",
    )
    gpr_parameters: GPRParameters = Field(
        default=None,
        title="gpr_parameters",
        description=" GPR parameters of GPR and its Kernel and Data of fitted GPR.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_type: MLModels = MLModels.GPR

    @classmethod
    def serialize(
        cls,
        model: CustomGPR,
        dt: Union[float, int],
        input: dict[str, Feature],
        output: dict[str, OutputFeature],
        training_info: Optional[dict] = None,
    ):
        """

        Args:
            model:    GaussianProcessRegressor from ScikitLearn.
            dt:     The length of time step of one prediction of GPR in seconds.
            input:  GPR input variables with their lag order.
            output: GPR output variables (which are automatically also inputs, as
                    we need them recursively in MPC.) with their lag order.
            training_info: Config of Trainer Class, which trained the Model.

        Returns:
            SerializedGPR version of the passed GPR.
        """
        if not all(
            hasattr(model, attr)
            for attr in ["kernel_", "alpha_", "L_", "X_train_", "y_train_"]
        ):
            raise ValueError(
                "To serialize a GPR, a fitted GPR must be passed, "
                "but an unfitted GPR has been passed here."
            )
        kernel_parameters = GPRKernelParameters.from_model(model)
        gpr_parameters = GPRParameters.from_model(model)
        return cls(
            dt=dt,
            input=input,
            output=output,
            data_handling=model.data_handling,
            kernel_parameters=kernel_parameters,
            gpr_parameters=gpr_parameters,
            trainer_config=training_info,
        )

    def deserialize(self) -> CustomGPR:
        """
        Deserializes SerializedGPR object and returns a scikit learn GaussionProcessRegressor.
        Returns:
            gpr_fitted: GPR version of the SerializedGPR
        """
        # Create unfitted GPR with standard Kernel and standard Parameters and Hyperparameters.
        kernel = ConstantKernel() * RBF() + WhiteKernel()
        gpr_unfitted = CustomGPR(
            kernel=kernel,
            copy_X_train=False,
        )
        # make basic fit for GPR
        gpr_fitted = self._basic_fit(gpr=gpr_unfitted)
        # update kernel parameters
        gpr_fitted.kernel_.k1.k1.constant_value = self.kernel_parameters.constant_value
        gpr_fitted.kernel_.k1.k1.constant_value_bounds = (
            self.kernel_parameters.constant_value_bounds
        )
        gpr_fitted.kernel_.k1.k2.length_scale = self.kernel_parameters.length_scale
        gpr_fitted.kernel_.k1.k2.length_scale_bounds = (
            self.kernel_parameters.length_scale_bounds
        )
        gpr_fitted.kernel_.k2.noise_level = self.kernel_parameters.noise_level
        gpr_fitted.kernel_.k2.noise_level_bounds = (
            self.kernel_parameters.noise_level_bounds
        )
        gpr_fitted.kernel_.theta = np.array(self.kernel_parameters.theta)
        # update gpr_parameters
        gpr_fitted.L_ = np.array(self.gpr_parameters.L)
        gpr_fitted.X_train_ = np.array(self.gpr_parameters.X_train)
        gpr_fitted.y_train_ = np.array(self.gpr_parameters.y_train)
        gpr_fitted.alpha_ = np.array(self.gpr_parameters.alpha)
        gpr_fitted.n_features_in_ = np.array(self.gpr_parameters.n_features_in)
        gpr_fitted.log_marginal_likelihood_value_ = np.array(
            self.gpr_parameters.log_marginal_likelihood_value
        )
        # update data handling
        gpr_fitted.data_handling.normalize = self.data_handling.normalize
        gpr_fitted.data_handling.scale = self.data_handling.scale
        if self.data_handling.mean:
            gpr_fitted.data_handling.mean = np.array(self.data_handling.mean)
        if self.data_handling.std:
            gpr_fitted.data_handling.std = np.array(self.data_handling.std)
        return gpr_fitted

    def _basic_fit(self, gpr: GaussianProcessRegressor):
        """
        Runs an easy fit to be able to populate with kernel_parameters and gpr_parameters
        afterward and therefore really fit it.
        Args:
            gpr: Unfitted GPR to fit
        Returns:
            gpr: fitted GPR
        """
        x = np.ones((1, len(self.input)))
        y = np.ones((1, len(self.output)))
        gpr.fit(
            X=x,
            y=y,
        )
        return gpr


class LinRegParameters(BaseModel):
    coef: list = Field(
        title="coefficients",
        description="Estimated coefficients for the linear regression problem. If multiple targets are passed during the fit (y 2D), this is a 2D array of shape (n_targets, n_features), while if only one target is passed, this is a 1D array of length n_features.",
    )
    intercept: Union[float, list] = Field(
        title="intercept",
        description="Independent term in the linear model. Set to 0.0 if fit_intercept = False.",
    )
    n_features_in: int = Field(
        title="number of input features",
        description="Number of features seen during fit.",
    )
    rank: int = Field(
        title="rank",
        description="Rank of matrix X. Only available when X is dense.",
    )
    singular: list = Field(
        title="singular",
        description="Singular values of X. Only available when X is dense.",
    )


class SerializedLinReg(SerializedMLModel):
    """
    Contains scikit-learn LinearRegression and provides functions to transform
    these to SerializedLinReg objects and vice versa.

    Attributes:

    """

    parameters: LinRegParameters = Field(
        title="parameters",
        description="Parameters of kernel of the fitted linear model.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_type: MLModels = MLModels.LINREG

    @classmethod
    def serialize(
        cls,
        model: LinearRegression,
        dt: Union[float, int],
        input: dict[str, Feature],
        output: dict[str, OutputFeature],
        training_info: Optional[dict] = None,
    ):
        """

        Args:
            model:    LinearRegression from ScikitLearn.
            dt:     The length of time step of one prediction of LinReg in seconds.
            input:  LinReg input variables with their lag order.
            output: LinReg output variables (which are automatically also inputs, as "
                    "we need them recursively in MPC.) with their lag order.
            training_info: Config of Trainer Class, which trained the Model.

        Returns:
            SerializedLinReg version of the passed linear model.
        """
        if not all(
            hasattr(model, attr)
            for attr in ["coef_", "intercept_", "n_features_in_", "rank_", "singular_"]
        ):
            raise ValueError(
                "To serialize a GPR, a fitted GPR must be passed, "
                "but an unfitted GPR has been passed here."
            )
        parameters = {
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "n_features_in": model.n_features_in_,
            "rank": model.rank_,
            "singular": model.singular_.tolist(),
        }
        parameters = LinRegParameters(**parameters)
        return cls(
            dt=dt,
            input=input,
            output=output,
            parameters=parameters,
            trainer_config=training_info,
        )

    def deserialize(self) -> LinearRegression:
        """
        Deserializes SerializedLinReg object and returns a LinearRegression object of scikit-learn.
        Returns:
            linear_model_fitted: LinearRegression version of the SerializedLinReg
        """
        linear_model_unfitted = LinearRegression()
        linear_model_fitted = self._basic_fit(linear_model=linear_model_unfitted)
        # update parameters
        linear_model_fitted.coef_ = np.array(self.parameters.coef)
        linear_model_fitted.intercept_ = np.array(self.parameters.intercept)
        linear_model_fitted.n_features_in_ = self.parameters.n_features_in
        linear_model_fitted.rank_ = self.parameters.rank
        linear_model_fitted.singular_ = np.array(self.parameters.singular)
        return linear_model_fitted

    def _basic_fit(self, linear_model: LinearRegression):
        """
        Runs an easy fit to be able to populate with parameters and gpr_parameters
        afterward and therefore really fit it.
        Args:
            linear_model: Unfitted linear model to fit.
        Returns:
            linear_model: fitted linear model.
        """
        x = np.ones((1, len(self.input)))
        y = np.ones((1, len(self.output)))
        linear_model.fit(
            X=x,
            y=y,
        )
        return linear_model


class SerializedKerasANN(SerializedMLModel):
    """
    Allows using a saved Keras ANN model (Sequential or Functional) in agentlib_mpc directly without serialization.

    attributes:
        model_path: Path to saved Keras ANN model.
    """

    model_path: Path = Field(
        default=None, description="Path, where the Keras ANN model is saved."
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_type: MLModels = MLModels.KerasANN

    @classmethod
    def serialize(
        cls,
        model: Union[Sequential, Functional],
        dt: Union[float, int],
        input: dict[str, Feature],
        output: dict[str, OutputFeature],
        training_info: Optional[dict] = None,
    ):
        """Serializes path to Keras Sequential or Functional ANN and returns SerializedKerasANN object"""

        try:
            model_path = model.save_path
        except AttributeError:
            model_path = Path("stored_models/model.keras")  # default value

        directory = Path(model_path).parent
        directory.mkdir(exist_ok=True)
        model.save(model_path)

        return cls(
            model_path=model_path,
            dt=dt,
            input=input,
            output=output,
            training_info=training_info,
        )

    def deserialize(self) -> Union[Sequential, Functional]:
        """Deserializes SerializedKerasANN object and returns a Keras Sequential ANN."""
        import keras

        ann = keras.saving.load_model(self.model_path)
        return ann


serialized_models = {
    MLModels.ANN: SerializedANN,
    MLModels.GPR: SerializedGPR,
    MLModels.LINREG: SerializedLinReg,
    MLModels.KerasANN: SerializedKerasANN,
}
