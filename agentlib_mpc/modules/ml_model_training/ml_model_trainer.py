import abc
import logging
import math
from pathlib import Path
from typing import Type, TYPE_CHECKING, Optional, List, Union
import json

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
from agentlib_mpc.utils.balancing import filter_time_series, filter_training_samples, remove_gaps
from agentlib_mpc.utils.balancing import FilteringMethodTimeSeries, FilteringMethodTrainingSamples
from agentlib_mpc.models.serialized_ml_model import (
    SerializedMLModel,
    SerializedANN,
    SerializedGPR,
    SerializedLinReg,
)
from agentlib_mpc.models.serialized_ml_model import CustomGPR, MLModels
from agentlib_mpc.data_structures import ml_model_datatypes
from agentlib_mpc.data_structures.interpolation import InterpolationMethods
from agentlib_mpc.utils.ml_model_eval_plotting import evaluate_model, plot_model_evaluation
from agentlib_mpc.utils.sampling import sample_values_to_target_grid
from pydantic import BaseModel, Field, field_validator
from keras import Sequential


logger = logging.getLogger(__name__)


class OnlineLearning(BaseModel):
    """Configuration for online learning capabilities."""
    active: bool = False
    training_at: float = float("inf")
    initial_ml_model_path: Optional[Path] = None

    @field_validator('training_at')
    @classmethod
    def validate_training_at(cls, v, info):
        if info.data.get('active', True) and v <= 0:
            raise ValueError("When online learning is active, training_at must be positive")
        return v

    @field_validator('initial_ml_model_path')
    @classmethod
    def validate_initial_model_path(cls, v, info):
        if v is not None and not v.exists():
            raise ValueError(f"Initial ML model path does not exist: {v}")
        return v

class MLModelTrainerConfig(BaseModuleConfig, abc.ABC):
    """
    Abstract Base Class for all Trainer Configs.
    """

    step_size: float
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
    number_of_training_repetitions: int = pydantic.Field(
        default=1,
        despription="Number of repeated trainings for the initial training with the same configuration to avoid random initialisation"
    )
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
    online_learning: OnlineLearning = Field(
        default_factory=OnlineLearning,
        description="Configuration for online learning capabilities."
    )
    filter_time_series: Optional[FilteringMethodTimeSeries] = pydantic.Field(
        default=None,
        description="Method to filter the time series data before training. If None, no filtering is applied."
    )
    filter_training_samples: Optional[FilteringMethodTrainingSamples] = pydantic.Field(
        default=None,
        description="Method to filter the training samples before training. If None, no filtering is applied."
    )

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
    
    @pydantic.field_validator("filter_time_series", mode="before")
    @classmethod
    def validate_filter_time_series(cls, filter_time_series, info: FieldValidationInfo):
        if filter_time_series is None:
            return filter_time_series
        if isinstance(filter_time_series, FilteringMethodTimeSeries):
            return filter_time_series
        # Convert string to enum
        allowed_methods = [methodname for methodname in FilteringMethodTimeSeries.__members__]
        if filter_time_series not in allowed_methods:
            raise ConfigurationError(
                f"filter_time_series '{filter_time_series}' is not a valid filtering method. "
                f"Choose from {allowed_methods} or set to None."
            )
        return FilteringMethodTimeSeries[filter_time_series]
    
    @pydantic.field_validator("filter_training_samples", mode="before")
    @classmethod
    def validate_filter_training_samples(cls, filter_training_samples, info: FieldValidationInfo):
        if filter_training_samples is None:
            return filter_training_samples
        if isinstance(filter_training_samples, FilteringMethodTrainingSamples):
            return filter_training_samples
        # Convert string to enum
        allowed_methods = [methodname for methodname in FilteringMethodTrainingSamples.__members__]
        if filter_training_samples not in allowed_methods:
            raise ConfigurationError(
                f"filter_training_samples '{filter_training_samples}' is not a valid filtering method. "
                f"Choose from {allowed_methods} or set to None."
            )
        return FilteringMethodTrainingSamples[filter_training_samples]


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
        self._data_sources: dict[str, Source] = {var: None for var in self.time_series_data.columns}
        self.ml_model_path = None
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
        yield self.env.timeout(self.config.online_learning.training_at)
        self._update_time_series_data()
        serialized_ml_model, best_model_path, training_data = self.retrain_model()
        self._update_ml_mpc_config(serialized_ml_model)
        if self.config.online_learning.active:
            while True:
                yield self.env.timeout(self.config.online_learning.training_at)
                self._update_time_series_data()
                serialized_ml_model, best_model_path, training_data = self.retrain_model()
                self._update_ml_mpc_config(serialized_ml_model)

    def _initialize_time_series_data(self) -> pd.DataFrame:
        """Loads simulation data to initialize the time_series data"""
        # Get all feature names from inputs and outputs in config
        feature_names = [var.name for var in self.config.inputs + self.config.outputs]
        time_series_data = {name: pd.Series(dtype=float) for name in feature_names}
        for ann_src in self.config.data_sources:
            loaded_time_series = load_sim(ann_src)
            for column in loaded_time_series.columns:
                if column in feature_names:
                    srs = loaded_time_series[column]
                    time_series_data[column] = pd.concat([time_series_data[column], srs])
        return pd.DataFrame(time_series_data)


    def retrain_model(self):
        """Trains the model based on the current historic data."""
        sampled = self.resample()
        inputs, outputs = self.create_inputs_and_outputs(sampled)
        training_data = self.divide_in_tvt(inputs, outputs)

        best_serialized_ml_model = None
        best_score = 0
        best_metrics = None
        i = 1

        self.ml_models = self.build_ml_model_sequence()

        for self.ml_model in self.ml_models:
            self.fit_ml_model(training_data)
            serialized_ml_model = self.serialize_ml_model()
            outputs = training_data.training_outputs.columns
            for name in outputs:
                total_score_mse, metrics_dict = evaluate_model(name, training_data, CasadiPredictor.from_serialized_model(serialized_ml_model))
                train_r2 = metrics_dict[name]["train_score_r2"]

                if abs(1 - train_r2) < abs(1 - best_score):
                    best_score = train_r2
                    best_serialized_ml_model = serialized_ml_model
                    best_metrics = metrics_dict
                    best_cross_check = total_score_mse

                path = Path(self.config.save_directory, self.agent_and_time, f"Iteration_{i}")
                if self.config.save_plots:
                    path.mkdir(parents=True, exist_ok=True)
                    plot_model_evaluation(
                        training_data,
                        name,
                        total_score_mse,
                        metrics_dict,
                        CasadiPredictor.from_serialized_model(serialized_ml_model),
                        show_plot=False,
                        save_path=path
                    )
                i += 1

        best_model_path = Path(self.config.save_directory, "best_model", self.agent_and_time)
        self.save_all(best_serialized_ml_model, training_data, best_model_path, name, best_metrics, best_cross_check)
        self.ml_model_path = best_model_path
        return best_serialized_ml_model, best_model_path, training_data

    def save_all(
            self,
            serialized_ml_model: SerializedMLModel,
            training_data: ml_model_datatypes.TrainingData,
            best_model_path,
            name,
            metrics_dict: dict,
            cross_check_score: float
    ):
        """Saves all relevant data and results of the training process if desired."""

        if self.config.save_data:
            training_data.save(best_model_path)

        if self.config.save_ml_model:
            self.save_ml_model(serialized_ml_model, path=best_model_path)

        plot_model_evaluation(
            training_data,
            name,
            cross_check_score,
            metrics_dict,
            CasadiPredictor.from_serialized_model(serialized_ml_model),
            show_plot=False,
            save_path=best_model_path
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

    def _update_time_series_data(self, sample_size: int = 100000):
        """
        Update Trainings Dataset with collected dara during simulation

        Args:
            sample_size: default=100000, If the amount of data exceeds sample_size, the data set is resampled

        Returns:
            Merged and resampled dataset for retraining
        """

        training_dataset = self.config.data_sources
        feature_names = list(self.history_dict.keys())
        dfs = []
        for path in training_dataset:
            var_names = pd.read_csv(path).iloc[0].values
            if any(feature in var_names for feature in feature_names):
                df = pd.read_csv(path, skiprows=2)
                df.columns = var_names
                columns_to_keep = [df.columns[0]] + [col for col in df.columns if col in feature_names]
                df = df[columns_to_keep]
                dfs.append(df)
        if dfs:
            trainings_data = pd.concat(dfs, ignore_index=True)
            trainings_data.set_index(trainings_data.columns[0], inplace=True)
        else:
            trainings_data = pd.DataFrame(columns=self.time_series_data.columns)

        # Check if data comes from severals sources
        df_list = []
        for feature_name, (time_stamps, values) in self.history_dict.items():
            temp_df = pd.DataFrame({feature_name: values}, index=time_stamps)

            duplicated_indices = temp_df.index.duplicated(keep=False)
            if duplicated_indices.any():
                for idx in temp_df.index[duplicated_indices].unique():
                    values_at_idx = temp_df.loc[idx, feature_name]
                    if isinstance(values_at_idx, pd.Series):
                        rounded_values = values_at_idx.round(3)
                        first_value = rounded_values.iloc[0]
                        if not all(v == first_value for v in rounded_values):
                            raise ValueError(f"Inconsistent values for feature '{feature_name}' at timestamp {idx}. "
                                             f"Values: {rounded_values.values}")
                temp_df = temp_df[~temp_df.index.duplicated(keep='first')]

            df_list.append(temp_df)

        # ratio of latest data to keep unfiltered
        KEEP_LATEST_RATIO = 0.0 # currently not used (0.0) due to filtering focusing on steady state removal only

        if df_list:
            data = pd.concat(df_list, axis=1).sort_index()

            # Get the first and last timestamps for overlap check
            first_timestamp = data.index[0]
            last_timestamp = data.index[-1]

            # Seperate latest part to not be filtered
            rows_to_filter = int(len(data) * (1 - KEEP_LATEST_RATIO))
            data_unfiltered = data.iloc[rows_to_filter:]
            data = data.iloc[:rows_to_filter]
        else:
            first_timestamp = last_timestamp = 0
            data = pd.DataFrame()

        for column in data.columns:
            if data[column].isnull().all():
                raise ValueError(f"No values for {column}. Check variables subscription")

        if trainings_data.empty:
            data = data.dropna(how='all')
            data = data.fillna(method='ffill').fillna(method='bfill')
            final_data = data
        else:
            # check overlap of simulated data and training data and shift training data if necessary
            if trainings_data.index[0] < last_timestamp and trainings_data.index[-1] > first_timestamp:
                trainings_data.index = trainings_data.index + (last_timestamp - trainings_data.index[0] + self.config.step_size)

            for column in data.columns:
                if data[column].isnull().any() and not data[column].isnull().all():
                    mask = data.index % self.config.step_size == 0
                    data_ts = data[mask]

                    if not data_ts[column].isnull().any():
                        data[column] = data[column].ffill().bfill()
            data = data.bfill()

            final_data = data.combine_first(trainings_data)

        # Filtering steady state parts of the time series
        WINDOW_SIZE = 100
        FILTERING_METHOD = self.config.filter_time_series
        threshold = 0.05


        # Set the default threshold based on the selected filtering method
        filter_name = (self.config.filter_time_series.name if self.config.filter_time_series else "").lower()
        match filter_name:
            case "roc":
                threshold = 0.05
            case "vol":
                threshold = 0.05
            case "knn":
                threshold = 0.000005

        if FILTERING_METHOD is not None:
            final_data = filter_time_series(final_data, method=FILTERING_METHOD, window_size=WINDOW_SIZE, threshold=threshold, verbose=True)
            final_data = remove_gaps(final_data, 2*60*60, verbose=True)

        final_data = pd.concat([final_data, data_unfiltered]).sort_index()

        # Apply sampling if data is too large
        if len(final_data) > sample_size:
            final_data = self.sample_data_max_variation(final_data, sample_size=sample_size) 
        self.time_series_data = final_data


    def sample_data_max_variation(self, data: pd.DataFrame, sample_size) -> pd.DataFrame:
        """
        Samples the time series data to maintain data quality and variety.

        Args:
            data: Input DataFrame with time series data
            sample_size: Target number of samples

        Returns:
            Sampled DataFrame maintaining data characteristics
        """

        # Ensure we keep the most recent data (last 20% of sample_size)
        recent_size = int(0.2 * sample_size)
        recent_data = data.iloc[-recent_size:]

        remaining_data = data.iloc[:-recent_size].copy()

        important_indices = set()
        for column in data.columns:
            series = remaining_data[column]

            important_indices.update([
                series.idxmin(),
                series.idxmax()
            ])

            rolling_std = series.rolling(window=5).std()
            high_variance_points = rolling_std.nlargest(int(sample_size * 0.1)).index
            important_indices.update(high_variance_points)

        remaining_size = sample_size - recent_size - len(important_indices)
        if remaining_size > 0:
            n_bins = 10
            bins = pd.qcut(remaining_data.index, n_bins, duplicates='drop')
            stratified_samples = []

            samples_per_bin = remaining_size // len(bins.categories)
            for bin_label in bins.categories:
                bin_data = remaining_data[bins == bin_label]
                if not bin_data.empty:
                    stratified_samples.append(bin_data.sample(
                        n=min(samples_per_bin, len(bin_data)),
                        random_state=42
                    ))

            sampled_data = pd.concat([
                remaining_data.loc[list(important_indices)],  # Important points
                pd.concat(stratified_samples),  # Stratified samples
                recent_data  # Recent data
            ]).sort_index()
        else:
            sampled_data = pd.concat([
                remaining_data.loc[list(important_indices)],
                recent_data
            ]).sort_index()

        return sampled_data


    def _update_ml_mpc_config(self, serialized_ml_model):
        ml_model_variable = AgentVariable(
            name=ml_model_datatypes.ML_MODEL_TO_MPC,
            alias=ml_model_datatypes.ML_MODEL_TO_MPC,
            value=serialized_ml_model,
            timestamp=self.env.time,
            source=self.source,
            shared=True
        )
        self.agent.data_broker.send_variable(
            variable=ml_model_variable,
            copy=False,
        )

    @abc.abstractmethod
    def build_ml_model(self):
        """
        Builds and returns an ml model
        """
        pass

    @abc.abstractmethod
    def build_ml_model_onlinelearning(self):
        """
         Returns a ml model from previous training
        """
        pass

    def build_ml_model_sequence(self):
        """
        Builds and returns an ml model sequence
        Build several trainer from same configuration and fit, to avoid bad results due to random initialisation
        """
        mls = list()
        n = self.config.number_of_training_repetitions
        if self.config.online_learning.active and (self.config.online_learning.initial_ml_model_path is not None or self.ml_model_path is not None):
            ml_model = self.build_ml_model_onlinelearning()
            mls.append(ml_model)
        else:
            for i in range(n):
                ml_model = self.build_ml_model()
                mls.append(ml_model)
        return mls

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
                f"Called ML Trainer in strict mode but there was insufficient data."
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
        ml_inputs, ml_outputs = self._define_features()

        serialized_ml = self.model_type.serialize(
            model=self.ml_model,
            dt=self.config.step_size,
            input=ml_inputs,
            output=ml_outputs,
            training_info=self.training_info,
        )
        return serialized_ml

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

        print(f"Total inputs before filtering has {len(inputs)} rows.")

        # calculate the sample count and shares
        num_of_samples = inputs.shape[0]
        n_training = max(int(self.config.train_share * num_of_samples),1)
        n_validation = n_training + max(int(self.config.validation_share * num_of_samples),1)

        # shuffle the data
        permutation = np.random.permutation(num_of_samples)
        inputs = inputs.iloc[permutation]
        outputs = outputs.iloc[permutation]

        # split the data
        training_inputs = inputs.iloc[0:n_training]
        training_outputs = outputs.iloc[0:n_training]

        
        print(f"Training inputs before filtering has {len(training_inputs)} rows.")

        # Apply balancing of the training set
        FILTERING_METHOD = self.config.filter_training_samples

        if FILTERING_METHOD is not None:
            # Adapt number of clusters to available samples (need at least 2 samples per cluster)
            n_clusters = min(10, max(2, len(training_inputs) // 2))
            training_inputs, training_outputs = filter_training_samples(
                df_inputs=training_inputs,
                df_targets=training_outputs,
                method=FILTERING_METHOD,
                threshold=n_clusters,
                verbose=True
            )

        validation_inputs = inputs.iloc[n_training:n_validation]
        validation_outputs = outputs.iloc[n_training:n_validation]
        test_inputs = inputs.iloc[n_validation:]
        test_outputs = outputs.iloc[n_validation:]

        return ml_model_datatypes.TrainingData(
            training_inputs=training_inputs,
            training_outputs=training_outputs,
            validation_inputs=validation_inputs,
            validation_outputs=validation_outputs,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
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
        ann.compile(loss="mse", optimizer="adam")
        return ann

    def build_ml_model_onlinelearning(self) -> Sequential:
        if hasattr(self, 'ml_model_path') and self.ml_model_path is not None:
            path = Path(self.ml_model_path) / "ml_model.json"
        else:
            path = self.config.online_learning.initial_ml_model_path

        if path is None:
            raise ValueError("No pre-trained model path provided for online learning.")

        ann = SerializedMLModel.load_serialized_model_from_file(path)
        keras_model = ann.deserialize()
        optimizer = ann.optimizer
        loss = ann.loss
        keras_model.compile(optimizer=optimizer, loss=loss)
        return keras_model

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

    def build_ml_model_onlinelearning(self):
        if hasattr(self, 'ml_model_path') and self.ml_model_path is not None:
            path = Path(self.ml_model_path) / "ml_model.json"
        else:
            path = self.config.online_learning.initial_ml_model_path

        if path is None:
            raise ValueError("No pre-trained model path provided for online learning.")

        serialized_model = SerializedMLModel.load_serialized_model_from_file(path)

        # Handle the case where the loaded model might not be a GPR model
        if not isinstance(serialized_model, SerializedGPR):
            raise TypeError(f"Expected SerializedGPR model but got {type(serialized_model)}")

        gpr = serialized_model.deserialize()

        # Configure data handling properties from the serialized model
        if hasattr(serialized_model, 'data_handling') and serialized_model.data_handling:
            if hasattr(serialized_model.data_handling, 'normalize'):
                gpr.data_handling.normalize = serialized_model.data_handling.normalize
            if hasattr(serialized_model.data_handling, 'scale'):
                gpr.data_handling.scale = serialized_model.data_handling.scale
            if hasattr(serialized_model.data_handling, 'mean'):
                gpr.data_handling.mean = serialized_model.data_handling.mean
            if hasattr(serialized_model.data_handling, 'std'):
                gpr.data_handling.std = serialized_model.data_handling.std

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

    def build_ml_model_onlinelearning(self):
        if hasattr(self, 'ml_model_path') and self.ml_model_path is not None:
            path = Path(self.ml_model_path) / "ml_model.json"
        else:
            path = self.config.online_learning.initial_ml_model_path

        if path is None:
            raise ValueError("No pre-trained model path provided for online learning.")

        serialized_model = SerializedMLModel.load_serialized_model_from_file(path)

        # Handle the case where the loaded model might not be a linear regression model
        if not isinstance(serialized_model, SerializedLinReg):
            raise TypeError(f"Expected SerializedLinReg model but got {type(serialized_model)}")

        linear_model = serialized_model.deserialize()
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
