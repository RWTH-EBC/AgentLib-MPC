import abc
import json
import logging
import subprocess

from enum import Enum

# from keras import Sequential
from pathlib import Path
from pydantic import ConfigDict, Field, BaseModel

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RBF
# from sklearn.linear_model import LinearRegression
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import LinearRegression
    from keras import Sequential


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
        model: Union["Sequential", "GaussianProcessRegressor", "LinearRegression"],
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


try:
    from agentlib_mpc.models.serialized_ml_model.serialized_ann import SerializedANN
except ImportError:
    keras_avai
from agentlib_mpc.models.serialized_ml_model.serialized_gpr import SerializedGPR
from agentlib_mpc.models.serialized_ml_model.serialized_linreg import SerializedLinReg

serialized_models = {
    MLModels.ANN: SerializedANN,
    MLModels.GPR: SerializedGPR,
    MLModels.LINREG: SerializedLinReg,
}
