from agentlib import Agent
from agentlib.core.errors import OptionalDependencyError
from sklearn.linear_model import LinearRegression

from agentlib_mpc.data_structures import ml_model_datatypes
from agentlib_mpc.models.serialized_ml_model.serialized_linreg import SerializedLinReg
from agentlib_mpc.modules.ml_model_training.ml_model_trainer import (
    MLModelTrainerConfig,
    MLModelTrainer,
)


class LinRegTrainerConfig(MLModelTrainerConfig):
    """
    Pydantic data model for GPRTrainer configuration parser
    """


class LinRegTrainer(MLModelTrainer):
    """
    Module that generates ANNs based on received data.
    """

    config: LinRegTrainerConfig
    model_type = SerializedLinReg

    def __init__(self, config: dict, agent: Agent):
        super().__init__(config, agent)

    def build_ml_model(self) -> "LinearRegression":
        """Build a linear model."""
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError as err:
            raise OptionalDependencyError(
                dependency_install="scikit-learn",
                used_object="Linear Regression",
            ) from err

        linear_model = LinearRegression()
        return linear_model

    def fit_ml_model(self, training_data: ml_model_datatypes.TrainingData):
        """Fits linear model to training data"""
        self.ml_model.fit(
            X=training_data.training_inputs,
            y=training_data.training_outputs,
        )
