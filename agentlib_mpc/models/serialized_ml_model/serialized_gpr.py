from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from agentlib_mpc.data_structures.ml_model_datatypes import Feature, OutputFeature
from agentlib_mpc.models.serialized_ml_model.serialized_ml_model import (
    SerializedMLModel,
    MLModels,
)


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
        data_handling=GPRDataHandlingParameters(),
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
