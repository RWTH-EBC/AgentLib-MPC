import os
from typing import Dict, Optional, Tuple

import pandas as pd
from agentlib.core import (
    BaseModuleConfig,
    BaseModule,
    Agent,
    AgentVariable,
    Source,
)
from agentlib.core.errors import ConfigurationError
from pydantic import Field

from agentlib_mpc.data_structures import mpc_datamodels
from agentlib_mpc.modules.mpc import create_optimization_backend
from agentlib_mpc.optimization_backends.backend import (
    OptimizationBackendT,
)

AG_VAR_DICT = dict[str, AgentVariable]


class MHEConfig(BaseModuleConfig):
    """
    Pydantic data model for MPC configuration parser
    """

    optimization_backend: dict
    time_step: float = Field(
        default=60,
        ge=0,
        description="Time step of the MHE.",
    )
    horizon: int = Field(
        default=5,
        ge=0,
        description="Estimation horizon of the MHE.",
    )
    known_parameters: mpc_datamodels.MPCVariables = Field(
        default=[],
        description="List of known parameters of the MHE. They are "
        "constant over the horizon. Parameters not listed "
        "here will have their default from the model file.",
    )
    estimated_parameters: mpc_datamodels.MPCVariables = Field(
        default=[],
        description="List of unknown parameters of the MHE. They are "
        "constant over the horizon and will be estimated.",
    )
    known_inputs: mpc_datamodels.MPCVariables = Field(
        default=[],
        description="List of known input variables of the MHE. Includes "
        "controls, disturbances, setpoints, dynamic constraint boundaries etc.",
    )
    estimated_inputs: mpc_datamodels.MPCVariables = Field(
        default=[],
        description="List of unknown input variables of the MHE. Includes "
        "mainly disturbances.",
    )  # AgentVariables for the initial condition of states to be optimized
    states: mpc_datamodels.MHEVariables = Field(
        default=[],
        description="List of all differential states of the MHE.",
    )
    shared_variable_fields = []


class MHE(BaseModule):
    """
    A moving horizon estimator.
    """

    config_type = MHEConfig
    config: MHEConfig
    var_ref: mpc_datamodels.MHEVariableReference

    def __init__(self, config: dict, agent: Agent):
        """
        Constructor for model predictive controller (MPC).
        Args:
            config:  name of the module
            agent:   agent the module belongs to
        Configs:
            outputs (object):
            inputs (object):
            ts: time step in s
            n (int): prediction horizon
            nc (int): control horizon (default prediction horizon)
        """
        super().__init__(config=config, agent=agent)

        measured_states, weights_states = self._create_auxiliary_variables()
        self.measured_states: AG_VAR_DICT = measured_states
        self.weights_states: AG_VAR_DICT = weights_states

        # creates a reference of variables, which have to be kept track of in a
        # dataframe, to provide a past trajectory for the MHE
        self._history_var_names: set[str] = {
            v.name for v in self.config.known_inputs
        }.union(measured_states)

        self.history: pd.DataFrame = pd.DataFrame(columns=self._history_var_names)

        # construct the optimization problem
        try:
            self._init_optimization()
        except (RuntimeError, ValueError) as err:
            raise ConfigurationError(
                f"The optimization backend of Agent {self.source} could not "
                f"finish its setup!"
            ) from err

    def _setup_optimization_backend(self) -> OptimizationBackendT:
        """Performs the setup of the optimization_backend, keeps track of status"""
        self.init_status = mpc_datamodels.InitStatus.during_update
        opti_back = create_optimization_backend(
            self.config.optimization_backend, self.agent.id
        )
        disc_opts = opti_back.discretization_options
        disc_opts.prediction_horizon = self.config.horizon
        disc_opts.time_step = self.config.time_step
        return opti_back

    def _setup_var_ref(self) -> mpc_datamodels.MHEVariableReference:
        var_ref = mpc_datamodels.MHEVariableReference.from_config(self.config)
        var_ref.measured_states = list(self.measured_states)
        var_ref.weights_states = list(self.weights_states)
        return var_ref

    def _after_config_update(self):
        self._create_auxiliary_variables()
        self.var_ref = self._setup_var_ref()
        self.optimization_backend = self._setup_optimization_backend()
        self._init_optimization()
        self.init_status = mpc_datamodels.InitStatus.ready

    def _init_optimization(self):
        """Performs the setup of the optimization backend."""
        self.optimization_backend.setup_optimization(
            var_ref=self.var_ref,
        )
        self.logger.info("%s: Initialized optimization problem.", self.agent.id)

    def process(self):
        while True:
            cache_vars = self.collect_variables_for_optimization()
            solution = self.optimization_backend.solve(
                now=self.env.now, cache_vars=cache_vars
            )
            self._set_estimation(solution)
            self._remove_old_values_from_history()
            yield self.env.timeout(self.config.time_step)

    def _remove_old_values_from_history(self):
        """Clears the history of all entries that are older than current time minus
        horizon length."""
        backwards_horizon_seconds = self.config.horizon * self.config.time_step
        oldest_relevant_time = self.env.now - backwards_horizon_seconds
        filt = self.history.index >= oldest_relevant_time
        self.history = self.history[filt]

    def _set_estimation(self, results: pd.DataFrame):
        """Sets the estimated variables to the DataBroker."""

        # parameters are scalars defined at the beginning of the problem, so we send
        # the first value in the parameter trajectory
        for par in self.var_ref.estimated_parameters:
            par_val = results[par].iloc[0]
            self.set(par, par_val)

        # we want to know the most recent state value, so we send
        # the last value in the state trajectory
        for var in self.var_ref.states + self.var_ref.estimated_inputs:
            value = results[var].iloc[-1]
            self.set(var, value)

    def register_callbacks(self):
        """Registers callbacks which listen to the variables which have to be saved as
        time series. These callbacks save the values in the history for use in the
        optimization."""

        for inp in self.var_ref.known_inputs:
            var = self.get(inp)
            self.agent.data_broker.register_callback(
                alias=var.alias,
                source=var.source,
                callback=self._callback_hist_vars,
                name=var.name,
            )

        # registers callback which listens to alias/source of the state variable, but
        # gets the name of the measured state as parameter, to correctly save it in the
        # history
        for state, meas_state in zip(self.var_ref.states, self.var_ref.measured_states):
            var = self.get(state)
            self.agent.data_broker.register_callback(
                alias=var.alias,
                source=var.source,
                callback=self._callback_hist_vars,
                name=meas_state,
            )

    def collect_variables_for_optimization(
        self, var_ref: mpc_datamodels.MHEVariableReference = None
    ) -> Dict[str, AgentVariable]:
        """Gets all variables noted in the var ref and puts them in a flat
        dictionary. The MHE Version of this function has to perform some checks and
        lookups extra, since variables come from different sources, and some need to
        incorporate trajectories of past values."""
        if var_ref is None:
            var_ref = self.var_ref

        # first fetch all variables with get, that are in the config
        all_variables = {v: self.get(v) for v in var_ref.all_variables()}

        # then, collect the variables for the weights and measured states, that have
        # been generated and are not in the config
        for ms_name, ms_var in self.measured_states.items():
            all_variables[ms_name] = ms_var
        for w_name, w_var in self.weights_states.items():
            all_variables[w_name] = w_var

        # for values whose past trajectory is required in the optimization, set the
        # var value to that trajectory
        for hist_var in self._history_var_names:
            past_values = self.history[hist_var].dropna()
            if not any(past_values):
                # if the history of a variable is empty, fallback to the scalar value
                continue

            # create copy to not mess up scalar value of original variable in case
            # fallback is needed
            updated_var = all_variables[hist_var].copy(update={"value": past_values})
            all_variables[hist_var] = updated_var

        return all_variables

    def _callback_hist_vars(self, variable: AgentVariable, name: str):
        """Adds received measured inputs to the past trajectory."""
        self.history.loc[variable.timestamp, name] = variable.value

    def _create_auxiliary_variables(self) -> tuple[AG_VAR_DICT, AG_VAR_DICT]:
        """Creates variables holding the weights and measurements of the states"""
        states: mpc_datamodels.MHEVariables = self.config.states
        measured_states: dict[str, AgentVariable] = {}
        weights_states: dict[str, AgentVariable] = {}
        for state in states:
            include = state.dict(include={"unit", "description"})

            weight_name = "weight_" + state.name
            measurement_name = "measured_" + state.name

            weights_states[weight_name] = mpc_datamodels.MPCVariable(
                name=weight_name,
                value=state.weight,
                type="float",
                source=Source(module_id=self.id),
                **include,
            )
            measured_states[measurement_name] = mpc_datamodels.MPCVariable(
                name=measurement_name,
                value=pd.Series(state.value),
                type="pd.Series",
                source=state.source,
                **include,
            )
        self.weights_states = weights_states
        self.measured_states = measured_states
        return measured_states, weights_states

    def get_results(self) -> Optional[pd.DataFrame]:
        """Read the results that were saved from the optimization backend and
        returns them as Dataframe.

        Returns:
            (results, stats) tuple of Dataframes.
        """
        results_file = self.optimization_backend.results_file
        try:
            results, _ = self.read_results_file(results_file)
            return results
        except FileNotFoundError:
            self.logger.error("Results file %s was not found.", results_file)
            return None

    @staticmethod
    def read_results_file(results_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read the provided csv-file as an MPC results file.
        Args:
            results_file: File path

        Returns:
            results, stats
            results is the Dataframe with all inputs and outputs of the MPC
            optimizations.
            stats is the Dataframe with matching solver stats
        """
        stats_file = mpc_datamodels.stats_path(results_file)
        results = pd.read_csv(results_file, index_col=[0, 1], header=[0, 1])
        stats = pd.read_csv(stats_file, index_col=0)
        return results, stats

    def cleanup_results(self):
        results_file = self.optimization_backend.results_file
        if not results_file:
            return
        os.remove(results_file)
        os.remove(mpc_datamodels.stats_path(results_file))
