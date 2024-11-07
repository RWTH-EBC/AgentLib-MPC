from typing import Union, Optional

import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from sklearn.linear_model import LinearRegression

from agentlib_mpc.data_structures.ml_model_datatypes import Feature, OutputFeature
from agentlib_mpc.models.serialized_ml_model.serialized_ml_model import (
    SerializedMLModel,
    MLModels,
)


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
