import abc
import os
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Union, Callable, TypeVar, Optional

import pydantic
from agentlib.core.errors import ConfigurationError
from pydantic import ConfigDict
from pydantic_core.core_schema import FieldValidationInfo

from agentlib.utils import custom_injection
from agentlib.core import AgentVariable, Model
from agentlib_mpc.data_structures import mpc_datamodels
from agentlib_mpc.data_structures.mpc_datamodels import (
    DiscretizationOptions,
)
from agentlib_mpc.data_structures.mpc_datamodels import Results

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=Model)


class BackendConfig(pydantic.BaseModel):
    model: dict
    discretization_options: DiscretizationOptions
    name: Optional[str] = None
    results_file: Optional[Path] = pydantic.Field(default=None)
    save_results: Optional[bool] = pydantic.Field(validate_default=True, default=None)
    overwrite_result_file: Optional[bool] = pydantic.Field(
        default=False, validate_default=True
    )
    model_config = ConfigDict(extra="forbid")

    @pydantic.field_validator("results_file")
    @classmethod
    def check_csv(cls, file: Path):
        if not file.suffix == ".csv":
            raise ConfigurationError(
                f"Results filename has to be a 'csv' file. Got {file} instead."
            )
        return file

    @pydantic.field_validator("save_results")
    @classmethod
    def disable_results_if_no_file(cls, save_results: bool, info: FieldValidationInfo):
        if save_results is None:
            # if user did not specify if results should be saved, we save them if a
            # file is specified.
            return bool(info.data["results_file"])
        if save_results and info.data["results_file"] is None:
            raise ConfigurationError(
                "'save_results' was true, however there was no results file provided."
            )
        return save_results

    @pydantic.field_validator("overwrite_result_file")
    @classmethod
    def check_overwrite(cls, overwrite_result_file: bool, info: FieldValidationInfo):
        """Checks, whether the overwrite results sttings are valid, and deletes
        existing result files if applicable."""
        res_file = info.data.get("results_file")
        if res_file and info.data["save_results"]:
            if overwrite_result_file:
                try:
                    os.remove(res_file)
                    os.remove(mpc_datamodels.stats_path(res_file))
                except FileNotFoundError:
                    pass
            else:
                if os.path.isfile(info.data["results_file"]):
                    raise FileExistsError(
                        f"Results file {res_file} already exists and will not be "
                        f"overwritten automatically. Set 'overwrite_result_file' to "
                        f"True to enable automatic overwrite it."
                    )
        return overwrite_result_file


class OptimizationBackend(abc.ABC):
    """
    Base class for all optimization backends. OptimizationBackends are a
    plugin for the 'mpc' module. They provide means to setup and solve the
    underlying optimization problem of the MPC. They also can save data of
    the solutions.
    """

    _supported_models: dict[str, ModelT] = {}
    mpc_backend_parameters = ("time_step", "prediction_horizon")
    config_type = BackendConfig

    def __init__(self, config: dict):
        self.logger = logger
        self.config = self.config_type(**config)
        self.model: ModelT = self.model_from_config(self.config.model)
        self.var_ref: Optional[mpc_datamodels.VariableReference] = None
        self.cost_function: Optional[Callable] = None
        self.stats = {}
        self._created_file: bool = False  # flag if we checked the file location

    def register_logger(self, logger: logging.Logger):
        """Registers a logger, can be used to use the module logger"""
        self.logger = logger

    @abc.abstractmethod
    def setup_optimization(self, var_ref: mpc_datamodels.VariableReference):
        """
        Performs all necessary steps to make the ``solve`` method usable.

        Args:
            var_ref: Variable Reference that specifies the role of each model variable
                in the mpc
        """
        self.var_ref = var_ref

    @abc.abstractmethod
    def solve(
        self, now: Union[float, datetime], current_vars: Dict[str, AgentVariable]
    ) -> Results:
        """
        Solves the optimization problem given the current values of the
        corresponding AgentVariables and system time. The standardization of
        return values is a work in progress.

        Args:
            now: Current time used for interpolation of input trajectories.
            current_vars: Dict of AgentVariables holding the values relevant to
                the optimization problem. Keys are the names

        Returns:
            A dataframe with all optimization variables over their respective
            grids. Depending on discretization, can include many nan's, so care
            should be taken when using this, e.g. always use dropna() after
            accessing a column.

             Example:
                      variables   mDot | T_0 | slack_T
                 time
                 0                0.1  | 298 | nan
                 230              nan  | 297 | 3
                 470              nan  | 296 | 2
                 588              nan  | 295 | 1
                 700              0.05 | 294 | nan
                 930              nan  | 294 | 0.1


        """
        raise NotImplementedError(
            "The 'OptimizationBackend' class does not implement this because "
            "it is individual to the subclasses"
        )

    def update_discretization_options(self, opts: dict):
        """Updates the discretization options with the new dict."""
        self.config.discretization_options = (
            self.config.discretization_options.model_copy(update=opts)
        )
        self.setup_optimization(var_ref=self.var_ref)

    def model_from_config(self, model: dict):
        """Set the model to the backend."""
        model = model.copy()
        _type = model.pop("type")
        custom_cls = custom_injection(config=_type)
        model = custom_cls(**model)
        if not any(
            (
                isinstance(model, _supp_model)
                for _supp_model in self._supported_models.values()
            )
        ):
            raise TypeError(
                f"Given model is of type {type(model)} but "
                f"should be instance of one of:"
                f"{', '.join(list(self._supported_models.keys()))}"
            )
        return model

    def get_lags_per_variable(self) -> dict[str, float]:
        """Returns the name of variables which include lags and their lag in seconds.
        The MPC module can use this information to save relevant past data of lagged
        variables"""
        return {}

    def results_folder_exists(self) -> bool:
        """
        Checks if the results folder already exists, and if not, creates it with
        headers.
        """
        if self._created_file:
            return True

        if self.results_file_exists():
            # todo, this case is weird, as it is the mistake-append
            self._created_file = True
            return True

        # we only check the file location once to save system calls
        self.config.results_file.parent.mkdir(parents=True, exist_ok=True)
        self._created_file = True
        return False

    def results_file_exists(self) -> bool:
        """
        Checks if the results file already exists.
        """
        return self.config.results_file.is_file()

    def update_model_variables(self, current_vars: Dict[str, AgentVariable]):
        """
        Internal method to write current data_broker to model variables.
        Only update values, not other module_types.
        """
        for inp in current_vars.values():
            logger.debug(f"Updating model variable {inp.name}={inp.value}")
            self.model.set(name=inp.name, value=inp.value)


OptimizationBackendT = TypeVar("OptimizationBackendT", bound=OptimizationBackend)


class ADMMBackend(OptimizationBackend):
    """Base class for implementations of optimization backends for ADMM
    algorithms."""

    @property
    @abc.abstractmethod
    def coupling_grid(self) -> list[float]:
        """Returns the grid on which the coupling variables are discretized."""
        raise NotImplementedError
