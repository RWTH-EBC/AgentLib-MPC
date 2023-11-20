import abc
import logging
import math
from pathlib import Path
from typing import Type, TYPE_CHECKING

import numpy as np
import pandas as pd
import pydantic
from agentlib.core import (
    BaseModuleConfig,
    Agent,
    BaseModule,
    AgentVariables,
    AgentVariable,
    Source,
)
from agentlib.core.errors import ConfigurationError
from pydantic_core.core_schema import FieldValidationInfo

from agentlib_mpc.data_structures.ml_model_datatypes import name_with_lag
from agentlib_mpc.models.casadi_predictor import CasadiPredictor
from agentlib_mpc.utils.analysis import load_sim
from agentlib_mpc.models.serialized_ml_model import (
    SerializedMLModel,
    SerializedANN,
    SerializedGPR,
    SerializedLinReg,
)
from agentlib_mpc.models.serialized_ml_model import CustomGPR, MLModels
from agentlib_mpc.data_structures import ml_model_datatypes
from agentlib_mpc.data_structures.interpolation import InterpolationMethods
from agentlib_mpc.utils.plotting.ml_model_test import evaluate_model
from agentlib_mpc.utils.sampling import sample_values_to_target_grid

if TYPE_CHECKING:
    from keras import Sequential


logger = logging.getLogger(__name__)


class MLModelTrainerConfig(BaseModuleConfig, abc.ABC):
    """
    Abstract Base Class for all Trainer Configs.
    """

    step_size: float
    retrain_delay: float = pydantic.Field(
        default=10000000000,
        description="Time in seconds, after which retraining is triggered in regular"
        " intervals",
    )
    inputs: AgentVariables = pydantic.Field(
        default=[],
        description="Variables which are inputs of the ML Model that should be trained.",
    )
    outputs: AgentVariables = pydantic.Field(
        default=[],
        description="Variables which are outputs of the ML Model that should be trained.",
    )
    lags: dict[str, int] = pydantic.Field(
        default={},
        description="Dictionary specifying the lags of each input and output variable. "
        "If not specified, will be set to one.",
        validate_default=True,
    )
    output_types: dict[str, ml_model_datatypes.OutputType] = pydantic.Field(
        default={},
        description="Dictionary specifying the output types of output variables. "
        "If not specified, will be set to 'difference'.",
        validate_default=True,
    )
    interpolations: dict[str, InterpolationMethods] = pydantic.Field(
        default={},
        description="Dictionary specifying the interpolation types of output variables. "
        "If not specified, will be set to 'linear'.",
        validate_default=True,
    )
    recursive_outputs: dict[str, bool] = pydantic.Field(
        default={},
        description="Dictionary specifying whether output variables are recursive, i.e."
        " automatically appear as an input as well. If not specified, will"
        " be set to 'recursive'.",
        validate_default=True,
    )
    train_share: float = 0.7
    validation_share: float = 0.15
    test_share: float = 0.15
    save_directory: Path = pydantic.Field(
        default=None, description="Path, where created ML Models should be saved."
    )
    save_data: bool = pydantic.Field(
        default=False, description="Whether the training data should be saved."
    )
    save_ml_model: bool = pydantic.Field(
        default=False, description="Whether the created ML Models should be saved."
    )
    save_plots: bool = pydantic.Field(
        default=False,
        description="Whether a plot of the created ML Models performance should be saved.",
    )
    MLModel: AgentVariable = pydantic.Field(
        default=AgentVariable(name="MLModel", value=None),
        description="Serialized ML Model which can be sent to other Agents.",
    )
    time_series_memory_size: int = pydantic.Field(
        default=1_000_000_000,
        description="Maximum size of the data which is kept in memory for the ML Model "
        "training. If saved data exceeds this value, the oldest data is "
        "deleted.",
    )
    time_series_length: float = pydantic.Field(
        default=10 * 365 * 24 * 3600,
        description="Maximum time window of data which is kept for the ML Model training. If"
        " saved data is older than current time minus time_series_length, "
        "it will be deleted.",
    )
    use_values_for_incomplete_data: bool = pydantic.Field(
        default=False,
        description="Default False. If True, the values of inputs and outputs which are"
        " defined in the config will be used for training, in case historic"
        " data has not reached the trainer. If False, an Error will be "
        "raised when the data is not sufficient.",
    )
    data_sources: list[Path] = pydantic.Field(
        default=[],
        description="List of paths to time series data, which can be loaded on "
        "initialization of the agent.",
    )
    shared_variable_fields: list[str] = ["MLModel"]

    @pydantic.field_validator("train_share", "validation_share", "test_share")
    @classmethod
    def check_shares_amount_to_one(cls, current_share, info: FieldValidationInfo):
        """Makes sure, the shares amount to one."""
        shares = []
        if "train_share" in info.data:
            shares.append(info.data["train_share"])
        if "validation_share" in info.data:
            shares.append(info.data["validation_share"])
        if "test_share" in info.data:
            shares.append(info.data["test_share"])
        shares.append(current_share)
        if len(shares) == 3:
            if not math.isclose(sum(shares), 1, abs_tol=0.01):
                raise ConfigurationError(
                    f"Provided training, validation and testing shares do not equal "
                    f"one. Got {sum(shares):.2f} instead."
                )
        return current_share

    @pydantic.field_validator("lags")
    @classmethod
    def fill_lags(cls, lags, info: FieldValidationInfo):
        """Adds lag one to all unspecified lags."""
        all_features = {var.name for var in info.data["inputs"] + info.data["outputs"]}
        lag_to_var_diff = set(lags).difference(all_features)
        if lag_to_var_diff:
            raise ConfigurationError(
                f"Specified lags do not appear in variables. The following lags do not"
                f" appear in the inputs or outputs of the ML Model: '{lag_to_var_diff}'"
            )
        all_lags = {feat: 1 for feat in all_features}
        all_lags.update(lags)
        return all_lags

    @pydantic.field_validator("output_types")
    @classmethod
    def fill_output_types(cls, output_types, info: FieldValidationInfo):
        """Adds output type one to all unspecified output types."""
        output_names = {out.name for out in info.data["outputs"]}
        type_to_var_diff = set(output_types).difference(output_names)
        if type_to_var_diff:
            raise ConfigurationError(
                f"Specified outputs for output_types do not appear in variables. The "
                f"following lags do not appear in the inputs or outputs of the ML Model: "
                f"'{type_to_var_diff}'"
            )
        all_output_types = {feat: "absolute" for feat in output_names}
        all_output_types.update(output_types)
        return all_output_types

    @pydantic.field_validator("interpolations")
    @classmethod
    def fill_interpolations(cls, interpolations, info: FieldValidationInfo):
        """Adds interpolation method to all unspecified methods."""
        all_features = {var.name for var in info.data["inputs"] + info.data["outputs"]}
        interp_to_var_diff = set(interpolations).difference(all_features)
        if interp_to_var_diff:
            raise ConfigurationError(
                f"Specified outputs for output_types do not appear in variables. The "
                f"following features do not appear in the inputs or outputs of the ML Model: "
                f"'{interp_to_var_diff}'"
            )
        all_interp_methods = {feat: "linear" for feat in all_features}
        all_interp_methods.update(interpolations)
        return all_interp_methods

    @pydantic.field_validator("recursive_outputs")
    @classmethod
    def fill_recursive_outputs(cls, recursives, info: FieldValidationInfo):
        """Adds recursive flag to all unspecified outputs."""
        output_names = {var.name for var in info.data["outputs"]}
        recursives_to_var_diff = set(recursives).difference(output_names)
        if recursives_to_var_diff:
            raise ConfigurationError(
                f"Specified outputs for recursive_outputs do not appear in variables. The "
                f"following features do not appear in the inputs or outputs of the ML Model: "
                f"'{recursives_to_var_diff}'"
            )
        all_recursive_flags = {feat: True for feat in output_names}
        all_recursive_flags.update(recursives)
        return all_recursive_flags

    @pydantic.field_validator("data_sources")
    @classmethod
    def check_data_sources_exist(cls, data_sources: list[Path]):
        """Checks if all given data sources exist"""
        existing_data = []
        for data_src in data_sources:
            if data_src.exists():
                existing_data.append(data_src)
            else:
                logger.error(f"Given data source file {data_src} does not exist.")
        return existing_data

    @pydantic.field_validator("save_data", "save_ml_model")
    @classmethod
    def check_if_save_path_is_there(cls, save_on: bool, info: FieldValidationInfo):
        save_path = info.data["save_directory"]
        if save_path is None:
            raise ConfigurationError(
                "ML Model saving is on, but no save_directory was specified."
            )
        return save_on


class MLModelTrainer(BaseModule, abc.ABC):
    """
    Abstract Base Class for all Trainer classes.
    """

    config: MLModelTrainerConfig
    model_type: Type[SerializedMLModel]

    def __init__(self, config: dict, agent: Agent):
        """
        Constructor for model predictive controller (MPC).
        """
        super().__init__(config=config, agent=agent)
        self.time_series_data = self._initialize_time_series_data()
        history_type = dict[str, [tuple[list[float], list[float]]]]
        self.history_dict: history_type = {
            col: ([], []) for col in self.time_series_data.columns
        }
        self._data_sources: dict[str, Source] = {
            var: None for var in self.time_series_data.columns
        }
        self.ml_model = self.build_ml_model()
        self.input_features, self.output_features = self._define_features()

    @property
    def training_info(self) -> dict:
        """Returns a dict with relevant config parameters regarding the training."""
        # We exclude all fields of the Base Trainer, as its fields are with regard to
        # data handling etc., and other relevant things from base trainer are already
        # in the serialized model.
        # However, parameters from child classes are relevant to the training of that
        # model, and will be included
        exclude = set(MLModelTrainerConfig.model_fields)
        return self.config.model_dump(exclude=exclude)

    def register_callbacks(self):
        for feat in self.config.inputs + self.config.outputs:
            var = self.get(feat.name)
            self.agent.data_broker.register_callback(
                alias=var.alias,
                source=var.source,
                callback=self._callback_data,
                name=var.name,
            )

    def process(self):
        while True:
            yield self.env.timeout(self.config.retrain_delay)
            self._update_time_series_data()
            serialized_ml_model = self.retrain_model()
            self.set(self.config.MLModel.name, serialized_ml_model)

    def _initialize_time_series_data(self) -> pd.DataFrame:
        """Loads simulation data to initialize the time_series data"""
        feature_names = list(self.config.lags.keys())
        time_series_data = {name: pd.Series(dtype=float) for name in feature_names}
        for ann_src in self.config.data_sources:
            loaded_time_series = load_sim(ann_src)
            for column in loaded_time_series.columns:
                if column in feature_names:
                    srs = loaded_time_series[column]
                    time_series_data[column] = pd.concat(
                        [time_series_data[column], srs]
                    )

        return pd.DataFrame(time_series_data)

    def retrain_model(self):
        """Trains the model based on the current historic data."""
        sampled = self.resample()
        inputs, outputs = self.create_inputs_and_outputs(sampled)
        training_data = self.divide_in_tvt(inputs, outputs)
        self.fit_ml_model(training_data)
        serialized_ml_model = self.serialize_ml_model()
        self.save_all(serialized_ml_model, training_data)
        return serialized_ml_model

    def save_all(
        self,
        serialized_ml_model: SerializedMLModel,
        training_data: ml_model_datatypes.TrainingData,
    ):
        """Saves all relevant data and results of the training process if desired."""
        path = Path(self.config.save_directory, self.agent_and_time)
        if self.config.save_data:
            training_data.save(path)
        if self.config.save_ml_model:
            self.save_ml_model(serialized_ml_model, path=path)
        if self.config.save_plots:
            evaluate_model(
                training_data,
                CasadiPredictor.from_serialized_model(serialized_ml_model),
                save_path=path,
                show_plot=False,
            )

    def _callback_data(self, variable: AgentVariable, name: str):
        """Adds received measured inputs to the past trajectory."""
        # check that only data from the same source is used
        if self._data_sources[name] is None:
            self._data_sources[name] = variable.source
        elif self._data_sources[name] != variable.source:
            raise ValueError(
                f"The trainer module got data from different sources "
                f"({self._data_sources[name]}, {variable.source}). This is likely not "
                f"intended. Please specify the intended source in the trainer config."
            )

        time_list, value_list = self.history_dict[name]
        time_list.append(variable.timestamp)
        value_list.append(variable.value)
        self.logger.debug(
            f"Updated variable {name} with {variable.value} at {variable.timestamp} s."
        )

    def _update_time_series_data(self):
        """Clears the history of all entries that are older than current time minus
        horizon length."""
        df_list: list[pd.DataFrame] = []
        for feature_name, (time_stamps, values) in self.history_dict.items():
            df = pd.DataFrame({feature_name: values}, index=time_stamps)
            df_list.append(df)
        self.time_series_data = pd.concat(df_list, axis=1).sort_index()

        data = self.time_series_data
        if not data.size:
            return

        # delete rows based on how old the data is
        cut_off_time = self.env.now - self.config.time_series_length
        cut_off_index = data.index.get_indexer([cut_off_time], method="backfill")[0]
        data.drop(data.index[:cut_off_index], inplace=True)

        # delete rows if the memory usage is too high
        del_rows_at_once = 20  # currently hard-coded
        while data.memory_usage().sum() > self.config.time_series_memory_size:
            data.drop(data.index[:del_rows_at_once], inplace=True)

    @abc.abstractmethod
    def build_ml_model(self):
        """
        Builds and returns an ann model
        """
        pass

    @abc.abstractmethod
    def fit_ml_model(self, training_data: ml_model_datatypes.TrainingData):
        """
        Fits the ML Model with the training data.
        """
        pass

    def resample(self) -> pd.DataFrame:
        """Samples the available time_series data to the required step size."""
        source_grids = {
            col: self.time_series_data[col].dropna().index
            for col in self.time_series_data.columns
        }

        # check if data for all features is sufficient
        features_with_insufficient_data = []
        for feat_name in list(source_grids):
            if len(source_grids[feat_name]) < 5:
                del source_grids[feat_name]
                features_with_insufficient_data.append(feat_name)
        if (
            not self.config.use_values_for_incomplete_data
            and features_with_insufficient_data
        ):
            raise RuntimeError(
                f"Called ANN Trainer in strict mode but there was insufficient data."
                f" Features with insufficient data are: "
                f"{features_with_insufficient_data}"
            )

        # make target grid, which spans the maximum length, where data for every feature
        # is available
        start = max(sg[0] for sg in source_grids.values())
        stop = min(sg[-1] for sg in source_grids.values())
        target_grid = np.arange(start, stop, self.config.step_size)

        # perform interpolation for all features with sufficient length
        sampled = {}
        for name, sg in source_grids.items():
            single_sampled = sample_values_to_target_grid(
                values=self.time_series_data[name].dropna(),
                original_grid=sg,
                target_grid=target_grid,
                method=self.config.interpolations[name],
            )
            sampled[name] = single_sampled
        sampled_data = pd.DataFrame(sampled, index=target_grid)

        # pad data with fix values when data is incomplete
        if self.config.use_values_for_incomplete_data:
            length = len(target_grid)
            for feat_name in features_with_insufficient_data:
                sampled_data[feat_name] = [self.get(feat_name).value] * length

        return sampled_data

    def serialize_ml_model(self) -> SerializedMLModel:
        """
        Serializes the ML Model, sa that it can be saved
        as json file.
        Returns:
            SerializedMLModel version of the passed ML Model.
        """
        ann_inputs, ann_outputs = self._define_features()

        serialized_ann = self.model_type.serialize(
            model=self.ml_model,
            dt=self.config.step_size,
            input=ann_inputs,
            output=ann_outputs,
            training_info=self.training_info,
        )
        return serialized_ann

    def save_ml_model(self, serialized_ml_model: SerializedMLModel, path: Path):
        """Saves the ML Model in serialized format."""
        serialized_ml_model.save_serialized_model(path=Path(path, "ml_model.json"))

    def _define_features(
        self,
    ) -> tuple[
        dict[str, ml_model_datatypes.Feature],
        dict[str, ml_model_datatypes.OutputFeature],
    ]:
        """Defines dictionaries for all features of the ANN based on the inputs and
        outputs. This will also be the order, in which the serialized ann is exported"""
        ann_inputs = {}
        for name in self.input_names:
            ann_inputs[name] = ml_model_datatypes.Feature(
                name=name,
                lag=self.config.lags[name],
            )
        ann_outputs = {}
        for name in self.output_names:
            ann_outputs[name] = ml_model_datatypes.OutputFeature(
                name=name,
                lag=self.config.lags[name],
                output_type=self.config.output_types[name],
                recursive=self.config.recursive_outputs[name],
            )
        return ann_inputs, ann_outputs

    @property
    def agent_and_time(self) -> str:
        """A string that specifies id and time. Used to create save paths"""
        return f"{self.agent.id}_{self.id}_{self.env.now}"

    @property
    def input_names(self):
        return [inp.name for inp in self.config.inputs]

    @property
    def output_names(self):
        return [out.name for out in self.config.outputs]

    def create_inputs_and_outputs(
        self, full_data_sampled: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Creates extra columns in the data which contain the shifted time-series data
        which is lagged accordingly. Returns a tuple (input_data, output_data)"""
        # inputs are all inputs, plus recursive outputs with lag
        inps = [name_with_lag(v.name, 0) for v in self.config.inputs]
        inps.extend(
            [
                name_with_lag(v.name, 0)
                for v in self.config.outputs
                if self.config.recursive_outputs[v.name]
            ]
        )

        outs = [v.name for v in self.config.outputs]
        input_df = pd.DataFrame(columns=inps)
        output_df = pd.DataFrame(columns=(outs))

        # inputs
        for input_name in input_df.columns:
            lag: int = self.config.lags[input_name]
            for k in range(0, lag):
                name = name_with_lag(input_name, k)
                input_df[name] = full_data_sampled[input_name].shift(k)

        # output
        for output_name in output_df.columns:
            output_df[output_name] = self._create_output_column(
                name=output_name, column=full_data_sampled[output_name]
            )

        # some rows have nan now due to lags and output shift, we remove them
        na_rows = input_df.isna().any(axis=1) + output_df.isna().any(axis=1)
        input_df = input_df.loc[~na_rows]
        output_df = output_df.loc[~na_rows]

        # we have to make sure the columns are in consistent order, so the network is
        # trained in the same way, that its features are defined when exported
        columns_ordered = ml_model_datatypes.column_order(
            inputs=self.input_features, outputs=self.output_features
        )
        input_df = input_df[columns_ordered]

        return input_df, output_df

    def _create_output_column(self, name: str, column: pd.Series):
        """Creates an output column in the table for training data. Depending on
        whether the feature is recursive, or represents a time delta, some changes have
         to be made."""
        output_type = self.config.output_types[name]
        recursive = self.config.recursive_outputs[name]
        if not recursive:
            return column
        if output_type == ml_model_datatypes.OutputType.difference:
            return column.shift(-1) - column
        else:  # output_type == OutputType.absolute
            return column.shift(-1)

    def divide_in_tvt(
        self,
        inputs: pd.DataFrame,
        outputs: pd.DataFrame,
    ):
        """splits the samples into mpc, validating and testing sets"""

        # calculate the sample count and shares
        num_of_samples = inputs.shape[0]
        n_training = int(self.config.train_share * num_of_samples)
        n_validation = n_training + int(self.config.validation_share * num_of_samples)

        # shuffle the data
        permutation = np.random.permutation(num_of_samples)
        inputs = inputs.iloc[permutation]
        outputs = outputs.iloc[permutation]

        # split the data
        return ml_model_datatypes.TrainingData(
            training_inputs=inputs.iloc[0:n_training],
            training_outputs=outputs.iloc[0:n_training],
            validation_inputs=inputs.iloc[n_training:n_validation],
            validation_outputs=outputs.iloc[n_training:n_validation],
            test_inputs=inputs.iloc[n_validation:],
            test_outputs=outputs.iloc[n_validation:],
        )


class ANNTrainerConfig(MLModelTrainerConfig):
    """
    Pydantic data model for ANNTrainer configuration parser
    """

    epochs: int = 100
    batch_size: int = 100
    layers: list[tuple[int, ml_model_datatypes.Activation]] = pydantic.Field(
        default=[(16, "sigmoid")],
        description="Hidden layers which should be created for the ANN. An ANN always "
        "has a BatchNormalization Layer, and an Output Layer the size of "
        "the output dimensions. Additional hidden layers can be specified "
        "here as a list of tuples: "
        "(#neurons of layer, activation function).",
    )
    early_stopping: ml_model_datatypes.EarlyStoppingCallback = pydantic.Field(
        default=ml_model_datatypes.EarlyStoppingCallback(),
        description="Specification of the EarlyStopping Callback for training",
    )


class ANNTrainer(MLModelTrainer):
    """
    Module that generates ANNs based on received data.
    """

    config: ANNTrainerConfig
    model_type = SerializedANN

    def __init__(self, config: dict, agent: Agent):
        super().__init__(config, agent)

    def build_ml_model(self) -> Sequential:
        """Build an ANN with a one layer structure, can only create one ANN"""
        from keras import layers

        ann = Sequential()
        ann.add(layers.BatchNormalization(axis=1))
        for units, activation in self.config.layers:
            ann.add(layers.Dense(units=units, activation=activation))
        ann.add(layers.Dense(units=len(self.config.outputs), activation="linear"))
        ann.compile(loss="mse", optimizer="adam", metrics=["rmse"])
        return ann

    def fit_ml_model(self, training_data: ml_model_datatypes.TrainingData):
        callbacks = []
        if self.config.early_stopping.activate:
            callbacks.append(self.config.early_stopping.callback())

        self.ml_model.fit(
            x=training_data.training_inputs,
            y=training_data.training_outputs,
            validation_data=(
                training_data.validation_inputs,
                training_data.validation_outputs,
            ),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
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

    def build_ml_model(self):
        """Build a GPR with a constant Kernel in combination with a white kernel."""
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

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

    def build_ml_model(self):
        """Build a linear model."""
        from sklearn.linear_model import LinearRegression

        linear_model = LinearRegression()
        return linear_model

    def fit_ml_model(self, training_data: ml_model_datatypes.TrainingData):
        """Fits linear model to training data"""
        self.ml_model.fit(
            X=training_data.training_inputs,
            y=training_data.training_outputs,
        )


ml_model_trainer = {
    MLModels.ANN: ANNTrainer,
    MLModels.GPR: GPRTrainer,
    MLModels.LINREG: LinRegTrainer,
}
