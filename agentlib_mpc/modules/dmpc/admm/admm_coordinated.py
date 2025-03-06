"""Module implementing the coordinated ADMM module."""

from collections import namedtuple
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np

from agentlib import Agent, AgentVariable
from agentlib.core.datamodels import Source
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable, Results
from agentlib_mpc.modules.dmpc.coordinated_mpc import (
    CoordinatedMPC,
    CoordinatedMPCConfig,
)
from agentlib.utils.validators import convert_to_list
import agentlib_mpc.data_structures.coordinator_datatypes as cdt
import agentlib_mpc.data_structures.admm_datatypes as adt
from agentlib_mpc.optimization_backends.backend import ADMMBackend
from agentlib_mpc.modules.dmpc.admm.admm import ADMMConfig


coupInput = namedtuple("coup_input", ["mean", "lam"])


class CoordinatedADMMConfig(CoordinatedMPCConfig, ADMMConfig):
    """Configuration for CoordinatedADMM."""

    shared_variable_fields: list[str] = CoordinatedMPCConfig.default(
        "shared_variable_fields"
    ) + ADMMConfig.default("shared_variable_fields")


class CoordinatedADMM(CoordinatedMPC):
    """
    Module to implement an ADMM agent, which is guided by a coordinator.
    Only optimizes based on callbacks.
    """

    config: CoordinatedADMMConfig
    var_ref: (
        adt.VariableReference
    )  # Explicitly typing var_ref to help with type checking

    def __init__(self, *, config: dict, agent: Agent):
        self._initial_setup = True  # flag to check that we don't compile ipopt twice
        self._admm_variables: dict[str, AgentVariable] = {}
        super().__init__(config=config, agent=agent)
        self._alias_to_input_names = {}
        self._create_coupling_alias_to_name_mapping()

    def _setup_var_ref(self) -> adt.VariableReference:
        """Override to return the ADMM-specific variable reference."""
        return adt.VariableReference.from_config(self.config)

    def _setup_optimization_backend(self) -> ADMMBackend:
        """Override to set up ADMM-specific backend with coupling variables."""
        self._admm_variables = self._create_couplings()
        return super()._setup_optimization_backend()

    def _create_couplings(self) -> dict[str, MPCVariable]:
        """Map coupling variables based on already setup model"""
        # Map couplings:
        _admm_variables: dict[str, MPCVariable] = {}
        for coupling in self.config.couplings:
            coupling.source = Source(agent_id=self.agent.id)
            coupling.shared = True

            # Create new variables for each coupling:
            include = {"unit": coupling.unit, "description": coupling.description}
            coupling_entry = adt.CouplingEntry(name=coupling.name)
            alias = adt.coupling_alias(coupling.alias)
            _admm_variables[coupling_entry.multiplier] = MPCVariable(
                name=coupling_entry.multiplier,
                value=[0],
                type="list",
                source=Source(module_id=self.id),
                **include,
            )
            _admm_variables[coupling_entry.local] = MPCVariable(
                name=coupling_entry.local,
                value=convert_to_list(coupling.value),
                alias=alias,
                type="list",
                source=Source(agent_id=self.agent.id),
                shared=True,
                **include,
            )
            _admm_variables[coupling_entry.mean] = MPCVariable(
                name=coupling_entry.mean,
                type="list",
                source=Source(module_id=self.id),
                **include,
            )
            lag_val = coupling.value or np.nan_to_num(
                (coupling.ub + coupling.lb) / 2, posinf=1000, neginf=1000
            )
            _admm_variables[coupling_entry.lagged] = MPCVariable(
                name=coupling_entry.lagged,
                value=lag_val,
                source=Source(module_id=self.id),
                **include,
            )

        # Exchange variables
        for exchange_var in self.config.exchange:
            exchange_var.source = Source(agent_id=self.agent.id)
            exchange_var.shared = True

            # Create new variables for each exchange:
            include = {
                "unit": exchange_var.unit,
                "description": exchange_var.description,
            }

            exchange_entry = adt.ExchangeEntry(name=exchange_var.name)
            alias = adt.exchange_alias(exchange_var.alias)
            _admm_variables[exchange_entry.multiplier] = MPCVariable(
                name=exchange_entry.multiplier,
                value=[0],
                type="list",
                source=Source(module_id=self.id),
                **include,
            )
            _admm_variables[exchange_entry.local] = MPCVariable(
                name=exchange_entry.local,
                value=convert_to_list(exchange_var.value),
                alias=alias,
                type="list",
                source=Source(agent_id=self.agent.id),
                shared=True,
                **include,
            )
            _admm_variables[exchange_entry.mean_diff] = MPCVariable(
                name=exchange_entry.mean_diff,
                type="list",
                source=Source(module_id=self.id),
                **include,
            )
            lag_val = exchange_var.value or np.nan_to_num(
                (exchange_var.ub + exchange_var.lb) / 2, posinf=1000, neginf=1000
            )
            _admm_variables[exchange_entry.lagged] = MPCVariable(
                name=exchange_entry.lagged,
                value=lag_val,
                source=Source(module_id=self.id),
                **include,
            )

        return _admm_variables

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
        self._set_algorithm_parameters(options=options)
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
        """
        opt_inputs = self.collect_variables_for_optimization()
        opt_inputs[adt.PENALTY_FACTOR] = self.penalty_factor_var
        self._optimization_inputs = opt_inputs

    @property
    def penalty_factor_var(self) -> MPCVariable:
        return MPCVariable(name=adt.PENALTY_FACTOR, value=self.config.penalty_factor)

    def _create_coupling_alias_to_name_mapping(self):
        """
        creates a mapping of alias to the variable names for multiplier and
        global mean that the optimization backend recognizes
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

    def collect_variables_for_optimization(self, var_ref=None):
        """Gets all variables noted in the var ref and puts them in a flat dictionary."""
        if var_ref is None:
            var_ref = self.var_ref

        # config variables
        variables = {v: self.get(v) for v in var_ref.all_variables()}

        # Add ADMM coupling variables
        variables.update(self._admm_variables)

        # history variables
        for hist_var in getattr(self, "_lags_dict_seconds", {}):
            past_values = self.history[hist_var]
            if not past_values:
                # if the history of a variable is empty, fallback to the scalar value
                continue

            # create copy to not mess up scalar value of original variable in case
            # fallback is needed
            updated_var = variables[hist_var].copy(
                update={"value": pd.Series(past_values)}
            )
            variables[hist_var] = updated_var

        return {**variables, **getattr(self, "_internal_variables", {})}

    def optimize(self, variable: AgentVariable):
        """
        Performs the optimization given the mean trajectories and multipliers from the
        coordinator.
        Replies with the local optimal trajectories.
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

        # send optimizationData back to coordinator to signal finished
        # optimization. Select only trajectory where index is at least zero, to not
        # send lags
        cons_traj = {}
        exchange_traj = {}
        for coup in self.config.couplings:
            cons_traj[coup.alias] = self._result[coup.name]
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
        """
        # this check catches the case, where the agent was not alive / registered at
        # the start of the round and thus did not participate and has no result
        if self._result is not None:
            self.set_actuation(self._result)
        self._result = None

    def _set_algorithm_parameters(self, options: adt.ADMMParameters):
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

    def _initial_coupling_values(self) -> Tuple[Dict[str, list], Dict[str, list]]:
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

    def shift_trajectories(self):
        """Shifts algorithm specific trajectories."""
        # Implementation specific for ADMM if needed
        pass
