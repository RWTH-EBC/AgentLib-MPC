"""Holds the base class for MPCs."""

import os
from typing import Tuple, Dict, Optional, Union

import pandas as pd
from pydantic import Field, field_validator

from agentlib.core.datamodels import (
    AgentVariable,
)
from agentlib.core import Model, BaseModule, BaseModuleConfig, Agent
from agentlib.core.errors import OptionalDependencyError, ConfigurationError
from agentlib.utils import custom_injection
from pydantic_core.core_schema import FieldValidationInfo

from agentlib_mpc.data_structures.mpc_datamodels import (
    VariableReference,
    InitStatus,
    Results,
)
from agentlib_mpc.optimization_backends import backend_types, uninstalled_backend_types
from agentlib_mpc.optimization_backends.backend import (
    OptimizationBackend,
    OptimizationBackendT,
)
from agentlib_mpc.data_structures import mpc_datamodels
from agentlib_mpc.utils.analysis import load_mpc, load_mpc_stats


class BaseMPCConfig(BaseModuleConfig):
    """
    Pydantic data model for MPC configuration parser
    """

    # todo use config of optimization backend in annotation and create like modules
    optimization_backend: dict
    time_step: float = Field(
        default=60,
        ge=0,
        description="Time step of the MPC.",
    )
    prediction_horizon: int = Field(
        default=5,
        ge=0,
        description="Prediction horizon of the MPC.",
    )
    sampling_time: Optional[float] = Field(
        default=None,  # seconds
        description="Sampling interval for control steps. If None, will be the same as"
        " time step. Does not affect the discretization of the MPC, "
        "only the interval with which there will be optimization steps.",
        validate_default=True,
    )
    parameters: mpc_datamodels.MPCVariables = Field(
        default=[],
        description="List of model parameters of the MPC. They are "
        "constant over the horizon. Parameters not listed "
        "here will have their default from the model file.",
    )
    inputs: mpc_datamodels.MPCVariables = Field(
        default=[],
        description="List of all input variables of the MPC. Includes "
        "predictions for disturbances, set_points, dynamic "
        "constraint boundaries etc.",
    )
    outputs: mpc_datamodels.MPCVariables = Field(
        default=[], description="List of all shared outputs of the MPC. "
    )
    # AgentVariables for the controls to be optimized
    controls: mpc_datamodels.MPCVariables = Field(
        default=[], description="List of all control variables of the MPC. "
    )
    # AgentVariables for the initial condition of states to be optimized
    states: mpc_datamodels.MPCVariables = Field(
        default=[],
        description="List of all differential states of the MPC. The "
        "entries can define the boundaries and the source for the measurements",
    )
    set_outputs: bool = Field(
        default=False,
        description="Sets the full output time series to the data broker.",
    )
    shared_variable_fields: list[str] = ["outputs", "controls"]

    @field_validator("sampling_time")
    @classmethod
    def default_sampling_time(cls, samp_time, info: FieldValidationInfo):
        if samp_time is None:
            samp_time = info.data["time_step"]
        return samp_time


def create_optimization_backend(optimization_backend, agent_id):
    """Set up the optimization_backend"""
    optimization_backend = optimization_backend.copy()
    if "type" not in optimization_backend:
        raise KeyError(
            "Given model config does not contain key 'type' (type of the model)."
        )
    _type = optimization_backend.pop("type")
    optimization_backend["name"] = agent_id
    if isinstance(_type, dict):
        custom_cls = custom_injection(config=_type)
        backend = custom_cls(**optimization_backend)
    elif isinstance(_type, str):
        if _type in uninstalled_backend_types:
            raise OptionalDependencyError(
                dependency_name=_type,
                dependency_install=uninstalled_backend_types[_type],
            )
        if _type not in backend_types:
            raise TypeError(
                f"Given backend is not a valid internal optimization "
                f"backend. Supported backends are "
                f"{', '.join(list(backend_types.keys()))}"
            )
        backend = backend_types[_type](config=optimization_backend)
    else:
        raise TypeError(
            f"Error loading optimization backend. Config "
            f"'type' has to be either str or dict. Got "
            f"{type(_type)} instead. "
        )
    assert isinstance(backend, OptimizationBackend)
    return backend


class BaseMPC(BaseModule):
    """
    A model predictive controller.
    More info to follow.
    """

    config: BaseMPCConfig

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
        self.init_status = mpc_datamodels.InitStatus.pre_module_init
        super().__init__(config=config, agent=agent)

        # Check that module config and model variables match
        unassigned_model_variables = self.assert_mpc_variables_are_in_model()
        assert unassigned_model_variables["inputs"] == set(), (
            f"All model inputs must be declared in the MPC config. Model "
            f"variable(s) '{unassigned_model_variables['inputs']}' is/are free."
        )

    def _setup_optimization_backend(self) -> OptimizationBackendT:
        """Performs the setup of the optimization_backend, keeps track of status"""
        self.init_status = mpc_datamodels.InitStatus.during_update
        opti_back = create_optimization_backend(
            self.config.optimization_backend, self.agent.id
        )
        opti_back.register_logger(self.logger)
        disc_opts = opti_back.config.discretization_options
        disc_opts.prediction_horizon = self.config.prediction_horizon
        disc_opts.time_step = self.config.time_step
        return opti_back

    def _setup_var_ref(self) -> mpc_datamodels.VariableReferenceT:
        return VariableReference.from_config(self.config)

    def _after_config_update(self):
        self.var_ref: mpc_datamodels.VariableReferenceT = self._setup_var_ref()
        self.optimization_backend: OptimizationBackendT = (
            self._setup_optimization_backend()
        )
        self._init_optimization()
        self.init_status = mpc_datamodels.InitStatus.ready

    def assert_subset(self, mpc_names, model_names, message_head):
        """
        Helper function for assert assert_mpc_variables_are_in_model. Asserts
        the variables of the var_ref corresponding to ref_key are a subset of
        a list of names provided (usually obtained from the model) and prints
        out an error if false. Returns the portion of model_names that are
        not in the given var_ref.
        """
        assert set(mpc_names).issubset(model_names), (
            f"{message_head} of MPC {self.agent.id} are not contained in "
            f"model. Names must match. The following variables defined for the "
            f"MPC do not appear in the model: "
            f"'{set(mpc_names).difference(model_names)}'."
        )
        return set(model_names).difference(mpc_names)

    def assert_mpc_variables_are_in_model(self) -> dict[str, set[str]]:
        """
        Checks whether all variables of var_ref are contained in the model.
        Returns names of model variables not contained in the var_ref,
        sorted by keys: 'states', 'inputs', 'outputs', 'parameters'.
        """

        # arguments for validation function:
        # (key in var_ref, model names, str for head error message)
        args = [
            (
                "states",
                self.model.get_state_names(),
                "Differential variables / States",
            ),
            ("controls", self.model.get_input_names(), "Controls"),
            ("inputs", self.model.get_input_names(), "Inputs"),
            ("outputs", self.model.get_output_names(), "Outputs"),
            ("parameters", self.model.get_parameter_names(), "Parameters"),
        ]

        # perform validations and make a dictionary of unassigned variables
        unassigned_by_mpc_var = {
            key: self.assert_subset(self.var_ref.__dict__[key], names, message)
            for key, names, message in args
        }

        # fix unassigned values for inputs
        intersection_input = set(unassigned_by_mpc_var["controls"]).intersection(
            unassigned_by_mpc_var["inputs"]
        )

        # return dict should have model variables as keys, not mpc variables
        unassigned_by_model_var = {
            "states": unassigned_by_mpc_var["states"],
            "inputs": intersection_input,
            "outputs": unassigned_by_mpc_var["outputs"],
            "parameters": unassigned_by_mpc_var["parameters"],
        }

        return unassigned_by_model_var

    def collect_variables_for_optimization(
        self, var_ref: mpc_datamodels.VariableReference = None
    ) -> Dict[str, AgentVariable]:
        """Gets all variables noted in the var ref and puts them in a flat
        dictionary."""
        if var_ref is None:
            var_ref = self.var_ref
        return {v: self.get(v) for v in var_ref.all_variables()}

        # class AgVarDropin:
        #     ub: float
        #     lb: float
        #     value: Union[float, list, pd.Series]
        #     interpolation_method: InterpolationMethod

    def process(self):
        while True:
            self.do_step()
            yield self.env.timeout(self.config.time_step)

    def register_callbacks(self):
        """Registers the init_optimization callback to all parameters which
        cannot be changed without recreating the optimization problem."""
        for key in OptimizationBackend.mpc_backend_parameters:
            self.agent.data_broker.register_callback(
                alias=key, source=None, callback=self.re_init_optimization
            )

    def _init_optimization(self):
        """Performs the setup of the optimization backend."""
        try:
            self.optimization_backend.setup_optimization(var_ref=self.var_ref)
        except (RuntimeError, ValueError) as err:
            raise ConfigurationError(
                f"The optimization backend of Agent {self.source} could not "
                f"finish its setup!"
            ) from err
        self.logger.info("%s: Initialized optimization problem.", self.agent.id)

    def re_init_optimization(self, parameter: AgentVariable):
        """Re-initializes the optimization backend with new parameters."""
        self.optimization_backend.discretization_options[
            parameter.name
        ] = parameter.value
        self._init_optimization()

    @property
    def model(self) -> Model:
        """
        Getter for current simulation model

        Returns:
            agentlib.model: Current simulation model
        """
        return self.optimization_backend.model

    def pre_computation_hook(self):
        """
        This method is called in every computation step before the optimization starts.
        Overwrite this method in a derived subclass if you want to take some actions
        each time before the optimal control problem is solved.
        """
        pass

    def do_step(self):
        """
        Performs an MPC step.
        """
        if not self.init_status == InitStatus.ready:
            self.logger.warning("Skipping step, optimization_backend is not ready.")
            return

        self.pre_computation_hook()

        # get new values from data_broker
        updated_vars = self.collect_variables_for_optimization()

        # solve optimization problem with up-to-date values from data_broker
        result = self.optimization_backend.solve(self.env.time, updated_vars)

        # Set variables in data_broker
        self.set_actuation(result)
        self.set_output(result)

    def set_actuation(self, solution: Results):
        """Takes the solution from optimization backend and sends the first
        step to AgentVariables."""
        self.logger.info("Sending optimal control values to data_broker.")
        tolerance = 1e-5
        for control in self.var_ref.controls:
            ub = self.get(control).ub
            lb = self.get(control).lb
            # take the first entry of the control trajectory
            actuation = solution[control][0]
            # if variables only slightly breach boundaries, clip
            if ub < actuation < ub + tolerance:
                actuation = ub
            if lb - tolerance < actuation < lb:
                actuation = lb
            self.set(control, actuation)

    def set_output(self, solution: Results):
        """Takes the solution from optimization backend and sends it to AgentVariables."""
        # Output must be defined in the conig as "type"="pd.Series"
        if not self.config.set_outputs:
            return
        self.logger.info("Sending optimal output values to data_broker.")
        df = solution.df
        for output in self.var_ref.outputs:
            series = df.variable[output]
            self.set(output, series)

    def get_results(self) -> Optional[pd.DataFrame]:
        """Read the results that were saved from the optimization backend and
        returns them as Dataframe.

        Returns:
            (results, stats) tuple of Dataframes.
        """
        results_file = self.optimization_backend.config.results_file
        if results_file is None or not self.optimization_backend.config.save_results:
            self.logger.info("No results were saved .")
            return None
        try:
            result, stat = self.read_results_file(results_file)
            self.warn_for_missed_solves(stat)
            return result
        except FileNotFoundError:
            self.logger.error("Results file %s was not found.", results_file)
            return None

    def warn_for_missed_solves(self, stats: Optional[pd.DataFrame]):
        """
        Read the solver information from the optimization
        Returns:
            Warning if solver fails
        """
        if stats is None:
            return
        if stats["success"].all():
            return
        failures = ~stats["success"]
        failure_indices = failures[failures].index.tolist()
        self.logger.warning(
            f"Warning: There were failed optimizations at the following times: "
            f"{failure_indices}."
        )

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

        results = load_mpc(results_file)
        stats = load_mpc_stats(results_file)
        return results, stats

    def cleanup_results(self):
        results_file = self.optimization_backend.config.results_file
        if not results_file:
            return
        os.remove(results_file)
        os.remove(mpc_datamodels.stats_path(results_file))
