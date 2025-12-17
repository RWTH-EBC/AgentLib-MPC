import logging
import platform
from pathlib import Path
from typing import Type, Optional
import numpy as np
import casadi as ca
import pandas as pd
import pydantic
from agentlib.core.errors import ConfigurationError

from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable, stats_path
from agentlib_mpc.optimization_backends.casadi_.core import system
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationVariable,
    OptimizationParameter,
)
from agentlib_mpc.optimization_backends.casadi_.core.discretization import (
    DiscretizationT,
    Results,
)
from agentlib_mpc.optimization_backends.backend import (
    OptimizationBackend,
    BackendConfig,
)
from agentlib_mpc.models.casadi_model import (
    CasadiModel,
)
from agentlib_mpc.data_structures import mpc_datamodels
from agentlib_mpc.data_structures.casadi_utils import (
    CasadiDiscretizationOptions,
    SolverFactory,
    DiscretizationMethod,
    SolverOptions,
)
from agentlib_mpc.utils import sampling

logger = logging.getLogger(__name__)


class CasadiBackendConfig(BackendConfig):
    discretization_options: CasadiDiscretizationOptions = pydantic.Field(
        default_factory=CasadiDiscretizationOptions
    )
    solver: SolverOptions = pydantic.Field(default_factory=SolverOptions)
    build_batch_bat: Optional[Path] = pydantic.Field(
        default=None,
        description="Path to a batch file, which can compile C code on windows.",
    )
    do_jit: Optional[bool] = pydantic.Field(
        default=None,
        description="Boolean to turn JIT of the optimization problems on or off.",
        validate_default=True,
    )
    save_only_stats: bool = pydantic.Field(
        default=False,
        description="If results should be saved, setting this to True will only save"
        "the optimization statistics. May be useful for longer timespans,"
        "if the results with all predictions gets too large.",
    )

    @pydantic.field_validator("do_jit")
    @classmethod
    def validate_compile(cls, do_jit, info: pydantic.FieldValidationInfo):
        """Checks whether code compilation should be done."""

        # if we're on Linux, we cannot generate the code as of now
        if platform.system() == "Linux":
            if do_jit is True:
                raise NotImplementedError(
                    "C Code generation not implemented yet for linux."
                )
            # if not specified or False, we do not do jit
            return False

        # assume we're on Windows. If there is no batch file, we have to return False
        bat_file = info.data["build_batch_bat"]
        if bat_file is None:
            if do_jit is True:
                raise ConfigurationError(
                    "Cannot do C-Code generation on Windows without specifying a "
                    "proper batch file through the 'build_batch_bat' option."
                )
            return False

        # at this point we are on Windows and have a (hopefully) valid batch file
        if do_jit is None:
            # the user provided a batch file but no clear instruction. For backwards
            # compatibility, we will assume compilation is desired and return True
            return True

        # if both do_jit and the batch file are specified, we do not modify do_jit
        return do_jit


class CasADiBackend(OptimizationBackend):
    """
    OptimizationBackend for solving the optimization problem with CasADi.
    Requires the model to be a CasADi model.
    """

    system_type: Type[system.SystemT]
    system: system.SystemT
    discretization_types: dict[DiscretizationMethod, Type[DiscretizationT]]
    discretization: DiscretizationT
    _supported_models = {"CasadiModel": CasadiModel}
    config_type = CasadiBackendConfig

    def setup_optimization(self, var_ref: mpc_datamodels.VariableReference):
        """
        Performs all necessary steps to make the ``solve`` method usable.
        To do this, it calls several auxiliary functions. These functions can
        be overloaded to change the resulting optimization problem.

        Args:
            var_ref: class with variable name lists sorted by function in the mpc.
        """
        super().setup_optimization(var_ref=var_ref)
        self.reset_setup_attributes()

        # connect variable roles defined by the mpc module with the model
        self.system.initialize(model=self.model, var_ref=self.var_ref)
        solver_factory = SolverFactory(
            do_jit=self.config.do_jit,
            bat_file=self.config.build_batch_bat,
            name=self.config.name,
            options=self.config.solver,
            logger=self.logger,
        )
        self.discretization.initialize(
            system=self.system, solver_factory=solver_factory
        )

    def solve(self, now: float, current_vars: dict[str, MPCVariable]) -> Results:
        # collect and format inputs
        mpc_inputs = self._get_current_mpc_inputs(agent_variables=current_vars, now=now)
        full_results = self.discretization.solve(mpc_inputs)
        self.save_result_df(full_results, now)

        return full_results

    def _get_current_mpc_inputs(
        self, agent_variables: dict[str, MPCVariable], now: float
    ) -> dict[str, ca.DM]:
        """
        Reads the value from all received AgentVariables and performs the
        necessary expansion/interpolation of values onto the correct grid.

        Args:
            agent_variables: dictionary containing all AgentVariables from the
                var_ref, with names as keys.
            now: current time, used for interpolation of trajectory data

        Returns:
            dictionary with keys matching the required input for
                self._mpc_inputs_to_nlp_inputs()
        """

        def get_variable_boundaries(var: OptimizationVariable) -> dict[str, ca.DM]:
            """
            Gets boundaries and initial guesses for all optimization
            variables of denotation 'of'. Currently, initial guesses are used
            without shifting.

            Args:
                var: denotation matching the variable type that is gathered

            Returns:
                dict of the form {lb_<den>: ca.MX, ub_<den>: ca.MX, guess_<den>: ca.MX}
            """
            ref_list = var.ref_names
            input_map = var.input_map
            grid = self.discretization.grid(var)

            lower_bounds, upper_bounds = [], []
            for ref in ref_list:
                agent_variable = agent_variables[ref]
                ub = sampling.sample(
                    trajectory=agent_variable.ub,
                    grid=grid,
                    current=now,
                    method=agent_variable.interpolation_method,
                )
                upper_bounds.append(ub)
                lb = sampling.sample(
                    trajectory=agent_variable.lb,
                    grid=grid,
                    current=now,
                    method=agent_variable.interpolation_method,
                )
                lower_bounds.append(lb)

            boundaries = input_map(
                ub_ref=ca.horzcat(*upper_bounds).T, lb_ref=ca.horzcat(*lower_bounds).T
            )

            return boundaries

        def get_parameter_values(par: OptimizationParameter) -> dict[str, ca.DM]:
            """
            Gets values for all optimization parameters of denotation 'of'

            Args:
                par: denotation matching the variable type that is gathered

            Returns:
                dict of the form {<den>: ca.MX}
            """
            ref_list = par.ref_names
            input_map = par.add_default_values
            grid = self.discretization.grid(par)

            input_matrix = []
            for ref in ref_list:
                var = agent_variables[ref]
                value = var.value
                if value is None:
                    raise ValueError(
                        f"Input for variable {ref} is empty. "
                        f"Cannot solve optimization problem."
                    )
                try:
                    interpolation_method = var.interpolation_method
                except AttributeError as e:
                    # Catch the case where normal AgentVariables got mixed into the
                    # optimization input, possibly due to subclassing the MPC class
                    # and dynamically changing the MPC input
                    raise TypeError(
                        f"The variable {ref} does not have an interpolationmethod. All "
                        f"Variables used in MPC need to be of type MPCVariable "
                        f"(subclass of AgentVariable). This is likely caused by an "
                        f"error in a custom module."
                    ) from e
                input_matrix.append(
                    sampling.sample(
                        trajectory=value,
                        grid=grid,
                        current=now,
                        method=interpolation_method,
                    )
                )

            return input_map(ref=ca.horzcat(*input_matrix).T)

        mpc_inputs = {}

        for sys_par in self.system.parameters:
            sys_par_values = get_parameter_values(par=sys_par)
            mpc_inputs.update(sys_par_values)
        for sys_var in self.system.variables:
            sys_var_boundaries = get_variable_boundaries(var=sys_var)
            mpc_inputs.update(sys_var_boundaries)

        return mpc_inputs

    def reset_setup_attributes(self):
        """Cleans all attributes that are used for optimization setup."""
        self.system = self.system_type()
        opts = self.config.discretization_options
        method = opts.method
        self.discretization = self.discretization_types[method](options=opts)
        self.discretization.logger = self.logger

    def save_result_df(
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

        # Calculate objective values
        df = results.df
        objective_names, objective_values = self.approximate_objective(df)

        if not self.results_folder_exists():
            if not self.config.save_only_stats:
                results.write_columns(res_file)
            results.write_combined_stats_columns(stats_path(res_file), objective_names)

        with open(stats_path(res_file), "a") as f:
            f.writelines(
                results.combined_stats_line(str(now), objective_values, objective_names)
            )
        if self.config.save_only_stats:
            return
        df.index = list(map(lambda x: str((now, x)), df.index))
        df.to_csv(res_file, mode="a", header=False)

    def approximate_objective(self, results_df: pd.DataFrame):
        """Returns the approximate objective value of this MPC step."""
        objective = self.system.objective

        # approximate objective over multiple shooting grid. This introduces slight but usually unimportant errors when actual objective is calculated with a different discretization.
        grid = np.arange(
            0,
            self.config.discretization_options.prediction_horizon
            * (self.config.discretization_options.time_step + 1),
            self.config.discretization_options.time_step,
        )
        objective_values = objective.calculate_values(results_df, grid)
        objective_names = [obj.name for obj in objective.objectives] + ["total"]

        return objective_names, objective_values
