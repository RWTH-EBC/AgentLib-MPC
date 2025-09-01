"""Module implementing the coordinated ADMM module, which works together
with a coordinator."""

from collections import namedtuple
from typing import Dict, Optional, List
import pandas as pd
import pydantic

from agentlib_mpc.data_structures import mpc_datamodels
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from .admm import ADMM, ADMMConfig
from agentlib_mpc.modules.dmpc.employee import MiniEmployee, MiniEmployeeConfig
from agentlib.utils.validators import convert_to_list
import agentlib_mpc.data_structures.coordinator_datatypes as cdt
import agentlib_mpc.data_structures.admm_datatypes as adt
from agentlib.core import AgentVariable, Agent


coupInput = namedtuple("coup_input", ["mean", "lam"])


class CoordinatedADMMConfig(MiniEmployeeConfig, ADMMConfig):
    shared_variable_fields: list[str] = MiniEmployeeConfig.default(
        "shared_variable_fields"
    ) + ADMMConfig.default("shared_variable_fields")

    @pydantic.field_validator("couplings", "exchange")
    def couplings_should_have_values(cls, value: List[AgentVariable]):
        """Asserts that couplings and exchange have values, as they are needed for
        initial guess."""
        for var in value:
            if var.value is None:
                raise ValueError(
                    "Couplings and Exchange Variables should have a value, as it is "
                    "required for the initial guess."
                )
        return value


class CoordinatedADMM(MiniEmployee, ADMM):
    """
    Module to implement an ADMM agent, which is guided by a coordinator.
    Only optimizes based on callbacks.
    """

    config: CoordinatedADMMConfig

    def __init__(self, *, config: dict, agent: Agent):
        self._initial_setup = True  # flag to check that we don't compile ipopt twice
        super().__init__(config=config, agent=agent)
        self._optimization_inputs: Dict[str, AgentVariable] = {}
        self._create_coupling_alias_to_name_mapping()
        self._result: Optional[pd.DataFrame] = None

    def process(self):
        # send registration request to coordinator
        timeout = self.config.registration_interval

        while True:
            if not self._registered_coordinator:
                guesses, ex_guess = self._initial_coupling_values()
                answer = adt.AgentToCoordinator(
                    local_trajectory=guesses, local_exchange_trajectory=ex_guess
                )
                self.set(cdt.REGISTRATION_A2C, answer.to_json())
            yield self.env.timeout(timeout)

    def registration_callback(self, variable: AgentVariable):
        """callback for registration"""
        if self._registered_coordinator:
            # ignore if registration has already been done
            return

        self.logger.debug(
            f"receiving {variable.name}={variable.value} from {variable.source}"
        )
        # global parameters to define optimisation problem
        value = cdt.RegistrationMessage(**variable.value)
        if not value.agent_id == self.source.agent_id:
            return
        options = adt.ADMMParameters(**value.opts)
        self._set_admm_parameters(options=options)
        guesses, ex_guess = self._initial_coupling_values()
        answer = adt.AgentToCoordinator(
            local_trajectory=guesses, local_exchange_trajectory=ex_guess
        )

        self._registered_coordinator = variable.source
        self.set(cdt.REGISTRATION_A2C, answer.to_json())

    def _after_config_update(self):
        # use some hacks to set jit false for the first time this function is called
        if (
            self.config.optimization_backend.get("do_jit", False)
            and self._initial_setup
        ):
            do_jit = True
            self.config.optimization_backend["do_jit"] = False
        else:
            do_jit = False
        super()._after_config_update()
        if self._initial_setup:
            self.config.optimization_backend["do_jit"] = do_jit
            self._initial_setup = False

    def get_new_measurement(self):
        """
        Retrieve new measurement from relevant sensors
        Returns:

        """
        opt_inputs = self.collect_variables_for_optimization()
        opt_inputs[adt.PENALTY_FACTOR] = self.penalty_factor_var
        self._optimization_inputs = opt_inputs

    def _create_coupling_alias_to_name_mapping(self):
        """
        creates a mapping of alias to the variable names for multiplier and
        global mean that the optimization backend recognizes
        Returns:

        """
        alias_to_input_names = {}
        for coupling in self.var_ref.couplings:
            coup_variable = self.get(coupling.name)
            coup_in = coupInput(mean=coupling.mean, lam=coupling.multiplier)
            alias_to_input_names[coup_variable.alias] = coup_in
        for coupling in self.var_ref.exchange:
            coup_variable = self.get(coupling.name)
            coup_in = coupInput(mean=coupling.mean_diff, lam=coupling.multiplier)
            alias_to_input_names[coup_variable.alias] = coup_in
        self._alias_to_input_names = alias_to_input_names

    def optimize(self, variable: AgentVariable):
        """
        Performs the optimization given the mean trajectories and multipliers from the
        coordinator.
        Replies with the local optimal trajectories.
        Returns:

        """
        # unpack message
        updates = adt.CoordinatorToAgent.from_json(variable.value)
        if not updates.target == self.source.agent_id:
            return
        self.logger.debug("Received update from Coordinator.")

        # load mpc inputs and current coupling inputs of this iteration
        opt_inputs = self._optimization_inputs.copy()

        # add the coupling inputs of this iteration to the other mpc inputs
        for alias, multiplier in updates.multiplier.items():
            coup_in = self._alias_to_input_names[alias]
            opt_inputs[coup_in.lam] = MPCVariable(name=coup_in.lam, value=multiplier)
            opt_inputs[coup_in.mean] = MPCVariable(
                name=coup_in.mean, value=updates.mean_trajectory[alias]
            )
        for alias, multiplier in updates.exchange_multiplier.items():
            coup_in = self._alias_to_input_names[alias]
            opt_inputs[coup_in.lam] = MPCVariable(name=coup_in.lam, value=multiplier)
            opt_inputs[coup_in.mean] = MPCVariable(
                name=coup_in.mean, value=updates.mean_diff_trajectory[alias]
            )

        opt_inputs[adt.PENALTY_FACTOR].value = updates.penalty_parameter
        # perform optimization
        self._result = self.optimization_backend.solve(
            now=self._start_optimization_at, current_vars=opt_inputs
        )
        self._result_obtained = True

        # send optimizationData back to coordinator to signal finished
        # optimization. Select only trajectory where index is at least zero, to not
        # send lags
        cons_traj = {}
        exchange_traj = {}
        for coup in self.config.couplings:
            cons_traj[coup.alias] = self._result[
                coup.name
            ]  # we can serialize numpy now, maybe make this easier
        for exchange in self.config.exchange:
            exchange_traj[exchange.alias] = self._result[exchange.name]

        opt_return = adt.AgentToCoordinator(
            local_trajectory=cons_traj, local_exchange_trajectory=exchange_traj
        )
        self.logger.debug("Sent optimal solution.")
        self.set(name=cdt.OPTIMIZATION_A2C, value=opt_return.to_json())

    def _finish_optimization(self):
        """
        Finalize an iteration. Usually, this includes setting the actuation.
        Returns:

        """
        # this check catches the case, where the agent was not alive / registered at
        # the start of the round and thus did not participate and has no result
        # Since the finish-signal of the coordinator is broadcast, it will trigger this
        # function even if the agent did not participate in the optimization before
        if self._result is not None:
            self.set_actuation(self._result)
        self._result = None

    def _set_admm_parameters(self, options: adt.ADMMParameters):
        """Sets new admm parameters, re-initializes the optimization problem
        and returns an initial guess of the coupling variables."""

        # update the config with new parameters
        new_config_dict = self.config.model_dump()
        new_config_dict.update(
            {
                adt.PENALTY_FACTOR: options.penalty_factor,
                cdt.TIME_STEP: options.time_step,
                cdt.PREDICTION_HORIZON: options.prediction_horizon,
            }
        )
        self.config = new_config_dict
        self.logger.info("%s: Reinitialized optimization problem.", self.agent.id)

    def _initial_coupling_values(self) -> tuple[Dict[str, list], Dict[str, list]]:
        """Gets the initial coupling values with correct trajectory length."""
        grid_len = len(self.optimization_backend.coupling_grid)
        guesses = {}
        exchange_guesses = {}
        for var in self.config.couplings:
            val = convert_to_list(var.value)
            # this overrides more precise guesses, but is more stable
            guesses[var.alias] = [val[0]] * grid_len
        for var in self.config.exchange:
            val = convert_to_list(var.value)
            exchange_guesses[var.alias] = [val[0]] * grid_len
        return guesses, exchange_guesses

    def init_iteration_callback(self, variable: AgentVariable):
        """Callback that answers the coordinators init_iteration flag."""
        if self._registered_coordinator:
            super().init_iteration_callback(variable)

    def _store_predicted_states(self, solution: mpc_datamodels.Results):
        """Store predicted state trajectories for fallback functionality."""
        if not self.config.enable_state_fallback:
            return

        df = solution.df
        current_time = self._start_optimization_at

        for state_name in self.var_ref.states:
            state_trajectory = df["variable"][state_name]
            # Convert index (which starts at 0) to actual timestamps
            for idx, value in state_trajectory.items():
                prediction_time = current_time + idx
                self.predicted_states[state_name][prediction_time] = value
