import abc
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Union, Callable, TypeVar, Optional, get_type_hints

import pandas as pd
import pydantic
from agentlib.core.errors import ConfigurationError
from pydantic_core.core_schema import FieldValidationInfo

from agentlib.utils import custom_injection
from agentlib.core import AgentVariable, Model
from agentlib_mpc.data_structures import mpc_datamodels
from agentlib_mpc.data_structures.mpc_datamodels import (
    stats_path, Results,
)

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=Model)


class BackendConfig(pydantic.BaseModel):
    model: dict
    name: Optional[str] = None
    results_file: Optional[Path] = pydantic.Field(default=None)
    save_results: Optional[bool] = pydantic.Field(validate_default=True, default=None)

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


class OptimizationBackend(abc.ABC):
    """
    Base class for all optimization backends. OptimizationBackends are a
    plugin for the 'mpc' module. They provide means to setup and solve the
    underlying optimization problem of the MPC. They also can save data of
    the solutions.
    """

    _supported_models: dict[str, ModelT] = {}
    mpc_backend_parameters = ("time_step", "prediction_horizon")
    config: BackendConfig

    def __init__(self, config: dict):
        self.config = get_type_hints(type(self))["config"](**config)
        self.model: ModelT = self.model_from_config(self.config.model)
        self.var_ref: Optional[mpc_datamodels.VariableReference] = None
        self.cost_function: Optional[Callable] = None
        self.stats = {}
        self._created_file: bool = False  # flag if we checked the file location

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

    def save_results(
        self,
        results: Results,
        now: float = 0,
    ):
        """
        Save the results of `solve` into a dataframe at each time step.

        Example results dataframe:

        value_type               variable              ...     lower
        variable                      T_0   T_0_slack  ... T_0_slack mDot_0
        time_step                                      ...
        2         0.000000     298.160000         NaN  ...       NaN    NaN
                  101.431499   297.540944 -149.465942  ...      -inf    0.0
                  450.000000   295.779780 -147.704779  ...      -inf    0.0
                  798.568501   294.720770 -146.645769  ...      -inf    0.0
        Args:
            results:
            now:

        Returns:

        """
        if not self.config.save_results:
            return

        res_file = self.config.results_file
        if not self.results_file_exists():
            results.write_columns(res_file)
            results.write_stats_columns(stats_path(res_file))

        df = results.df
        df.index = list(map(lambda x: str((now, x)), df.index))
        with open(res_file, "a") as f:
            df.to_csv(f, mode="a", header=False)

        with open(stats_path(res_file), "a") as f:
            f.writelines(results.stats_line(str(now)))

    def get_lags_per_variable(self) -> dict[str, float]:
        """Returns the name of variables which include lags and their lag in seconds.
        The MPC module can use this information to save relevant past data of lagged
        variables"""
        return {}

    def results_file_exists(self) -> bool:
        """Checks if the results file already exists, and if not, creates it with
        headers."""
        if self._created_file:
            return True

        if self.config.results_file.is_file():
            # todo, this case is weird, as it is the mistake-append
            self._created_file = True
            return True

        # we only check the file location once to save system calls
        self.config.results_file.parent.mkdir(parents=True, exist_ok=True)
        self._created_file = True
        return False


OptimizationBackendT = TypeVar("OptimizationBackendT", bound=OptimizationBackend)


class ADMMBackend(OptimizationBackend):
    """Base class for implementations of optimization backends for ADMM
    algorithms."""

    def __init__(self, *args, **kwargs):
        super(ADMMBackend, self).__init__(*args, **kwargs)
        self.it: int = 0
        self.results: list[pd.DataFrame] = []
        self.now: float = 0
        self.result_stats: list[str] = []

    @property
    @abc.abstractmethod
    def coupling_grid(self) -> list[float]:
        """Returns the grid on which the coupling variables are discretized."""
        raise NotImplementedError

    def save_results(
        self,
        results: Results,
        now: float = 0,
    ):
        """
        Save the results of `solve` into a dataframe at each time step.

        Example results dataframe:

        value_type               variable              ...     lower
        variable                      T_0   T_0_slack  ... T_0_slack mDot_0
        time_step                                      ...
        2         0.000000     298.160000         NaN  ...       NaN    NaN
                  101.431499   297.540944 -149.465942  ...      -inf    0.0
                  450.000000   295.779780 -147.704779  ...      -inf    0.0
                  798.568501   294.720770 -146.645769  ...      -inf    0.0
        Args:
            results:
            now:

        Returns:

        """
        if not self.config.save_results:
            return

        res_file = self.config.results_file

        if self.results_file_exists():
            self.it += 1
            if now != self.now:  # means we advanced to next step
                self.it = 0
                self.now = now
        else:
            self.it = 0
            self.now = now
            results.write_columns(res_file)
            results.write_stats_columns(stats_path(res_file))

        df = results.df
        df.index = list(map(lambda x: str((now, self.it, x)), df.index))
        self.results.append(df)

        # append solve stats
        index = str((now, self.it))
        self.result_stats.append(results.stats_line(index))

        # save last results at the start of new sampling time, or if 1000 iterations
        # are exceeded
        if not (self.it == 0 or self.it % 1000 == 0):
            return

        with open(res_file, "a") as f:
            for iteration_result in self.results:
                # todo try saving numpy arrays
                iteration_result.to_csv(f, mode="a", header=False)

        with open(stats_path(res_file), "a") as f:
            f.writelines(self.result_stats)
        self.results = []
        self.result_stats = []
