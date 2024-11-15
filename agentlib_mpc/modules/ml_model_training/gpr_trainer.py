import numpy as np
import pydantic
from agentlib import Agent
from agentlib.core.errors import OptionalDependencyError

from agentlib_mpc.data_structures import ml_model_datatypes
from agentlib_mpc.models.serialized_ml_model.serialized_gpr import (
    SerializedGPR,
    CustomGPR,
)
from agentlib_mpc.modules.ml_model_training.ml_model_trainer import (
    MLModelTrainerConfig,
    MLModelTrainer,
    logger,
)


class GPRTrainerConfig(MLModelTrainerConfig):
    """
    Pydantic data model for GPRTrainer configuration parser
    """

    constant_value_bounds: tuple = (1e-3, 1e5)
    length_scale_bounds: tuple = (1e-3, 1e5)
    noise_level_bounds: tuple = (1e-3, 1e5)
    noise_level: float = 1.5
    normalize: bool = pydantic.Field(
        default=False,
        description="Defines whether the training data and the inputs are for prediction"
        "are normalized before given to GPR.",
    )
    scale: float = pydantic.Field(
        default=1.0,
        description="Defines by which value the output data is divided for training and "
        "multiplied after prediction.",
    )
    n_restarts_optimizer: int = pydantic.Field(
        default=0,
        description="Defines the number of restarts of the Optimizer for the "
        "gpr_parameters of the kernel.",
    )


class GPRTrainer(MLModelTrainer):
    """
    Module that generates ANNs based on received data.
    """

    config: GPRTrainerConfig
    model_type = SerializedGPR

    def __init__(self, config: dict, agent: Agent):
        super().__init__(config, agent)

    def build_ml_model(self) -> CustomGPR:
        """Build a GPR with a constant Kernel in combination with a white kernel."""
        try:
            from sklearn.gaussian_process.kernels import (
                ConstantKernel,
                RBF,
                WhiteKernel,
            )
        except ImportError as err:
            raise OptionalDependencyError(
                dependency_install="scikit-learn",
                used_object="Gaussian Process Regression",
            ) from err

        kernel = ConstantKernel(
            constant_value_bounds=self.config.constant_value_bounds
        ) * RBF(length_scale_bounds=self.config.length_scale_bounds) + WhiteKernel(
            noise_level=self.config.noise_level,
            noise_level_bounds=self.config.noise_level_bounds,
        )

        gpr = CustomGPR(
            kernel=kernel,
            copy_X_train=False,
            n_restarts_optimizer=self.config.n_restarts_optimizer,
        )
        gpr.data_handling.normalize = self.config.normalize
        gpr.data_handling.scale = self.config.scale
        return gpr

    def fit_ml_model(self, training_data: ml_model_datatypes.TrainingData):
        """Fits GPR to training data"""
        if self.config.normalize:
            x_train = self._normalize(training_data.training_inputs.to_numpy())
        else:
            x_train = training_data.training_inputs
        y_train = training_data.training_outputs / self.config.scale
        self.ml_model.fit(
            X=x_train,
            y=y_train,
        )

    def _normalize(self, x: np.ndarray):
        # update the normal and the mean
        mean = x.mean(axis=0, dtype=float)
        std = x.std(axis=0, dtype=float)
        for idx, val in enumerate(std):
            if val == 0:
                logger.info(
                    "Encountered zero while normalizing. Continuing with a std of one for this Input."
                )
                std[idx] = 1.0

        if mean is None and std is not None:
            raise ValueError("Please update std and mean.")

        # save mean and standard deviation to data_handling
        self.ml_model.data_handling.mean = mean.tolist()
        self.ml_model.data_handling.std = std.tolist()

        # normalize x and return
        return (x - mean) / std
