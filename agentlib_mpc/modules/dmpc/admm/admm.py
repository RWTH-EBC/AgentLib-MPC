"""Holds functionality for ADMM modules."""
import time
import threading
from typing import List, Dict, Tuple, Iterable, Optional, TypeVar, Union
import queue
from enum import Enum, auto

import numpy as np
import pandas as pd
from agentlib.core.errors import ConfigurationError
from pydantic import field_validator, Field

from agentlib.core import (
    Source,
    AgentVariable,
)

from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from agentlib_mpc.modules.dmpc import DistributedMPC, DistributedMPCConfig
from agentlib_mpc.optimization_backends.backend import ADMMBackend
from agentlib.utils.validators import convert_to_list, is_list_of_agent_variables
from agentlib_mpc.data_structures import mpc_datamodels
import agentlib_mpc.data_structures.admm_datatypes as adt
from agentlib_mpc.optimization_backends.casadi_.core.discretization import Results

# noinspection PyArgumentList
class ModuleStatus(Enum):
    not_started = auto()
    syncing = auto()
    at_registration = auto()
    optimizing = auto()
    updating = auto()
    waiting_for_other_agents = auto()
    sleeping = auto()


# noinspection PyArgumentList
class ParticipantStatus(Enum):
    not_participating = auto()
    available = auto()
    confirmed = auto()
    not_available = auto()


class ADMMParticipation:
    """Holds data for the status of a shared variable of another system."""

    def __init__(self, variable):
        self.variable: AgentVariable = variable
        self.status: ParticipantStatus = ParticipantStatus.not_participating
        # no more than two messages should stack
        self.received: queue.Queue = queue.Queue(maxsize=5)

    def empty_memory(self):
        while True:
            try:
                self.received.get_nowait()
            except queue.Empty:
                break

    def de_register(self):
        self.status = ParticipantStatus.not_participating
        self.empty_memory()


class ADMMConfig(DistributedMPCConfig):
    couplings: List[mpc_datamodels.MPCVariable] = []
    exchange: List[mpc_datamodels.MPCVariable] = []

    penalty_factor: float = Field(
        default=10,
        ge=0,
        description="Penalty factor of the ADMM algorithm. Should be equal for all "
        "agents.",
    )
    iteration_timeout: float = Field(
        default=20,
        ge=0,
        description="Maximum computation + waiting time for one iteration.",
    )
    registration_period: float = Field(
        default=2,
        ge=0,
        description="Time spent on registration before each optimization",
    )
    max_iterations: float = Field(
        default=20,
        ge=0,
        description="Maximum number of ADMM iterations before termination of control "
        "step.",
    )

    @field_validator(
        "exchange", "couplings", "parameters", "inputs", "outputs", "controls", "states"
    )
    @classmethod
    def check_prefixes_of_variables(cls, variables: list[AgentVariable]):
        """Ensures no user provided variable is named with the reserved ADMM prefix."""
        conf_err = ConfigurationError(
            f"Do not use variables that start with "
            f"'{adt.ADMM_PREFIX}' in an ADMM config."
        )
        for var in variables:
            if var.name.startswith(adt.ADMM_PREFIX):
                raise conf_err
        return variables


ADMMConfigT = TypeVar("ADMMConfigT", bound=ADMMConfig)


class ADMM(DistributedMPC):
    """
    This class represents a module participating in a fully decentralized
    Consensus-ADMM optimization for distributed MPC.
    Agents autonomously send the values of their coupling variables, register
    other participants and perform update steps.
    """

    config: ADMMConfig
    var_ref: adt.VariableReference

    def __init__(self, config: dict, agent):
        self.var_qu = queue.Queue()
        self.start_step = threading.Event()
        self._status: ModuleStatus = ModuleStatus.syncing
        self._registered_participants = {}
        self._admm_variables: dict[str, AgentVariable] = {}
        super().__init__(config=config, agent=agent)

    def collect_couplings_for_optimization(self):
        """Collects updated AgentVariables only of the coupling variables."""
        coup_vars = {}
        for coup in self.var_ref.couplings + self.var_ref.exchange:
            coup_vars.update(
                {v: self._admm_variables[v] for v in coup.admm_variables()}
            )
        coup_vars["penalty_factor"] = self.penalty_factor_var
        return coup_vars

    def process(self):
        # this thread will perform the optimization whenever start_step is set
        thread = threading.Thread(
            target=self._admm_loop, daemon=True, name=f"admm_loop_{self.agent.id}"
        )
        thread.start()
        self.agent.register_thread(thread=thread)

        self._status: ModuleStatus = ModuleStatus.syncing
        yield self._sync_start()
        self.logger.info("Starting periodic execution of admm algorithm")

        while True:
            self.start_step.set()
            yield self.env.timeout(self.config.time_step)

    def _sync_start(self):
        """Waits until time is a multiple of the time step."""
        time_step = self.config.time_step
        delta = time_step - (time.time() % time_step)
        wait_time = delta
        self.logger.info("Waiting %s s to sync admm algorithm", wait_time)
        return self.env.timeout(wait_time)

    def _admm_loop(self):
        """Triggers the optimization whenever self.start_step is set."""
        while True:
            self._status: ModuleStatus = ModuleStatus.sleeping
            self.start_step.wait()
            self.start_step.clear()
            self.admm_step()
            if self.start_step.isSet():
                self.logger.error(
                    "%s: Start of ADMM round was requested before "
                    "last one finished. Waiting until next "
                    "cycle."
                )
                self.start_step.clear()

    def admm_step(self):
        """Performs an entire ADMM optimization."""

        self._perform_registration()

        # get optimization inputs
        self._set_mean_coupling_values()
        opt_inputs = self.collect_variables_for_optimization()
        self.pre_computation_hook()

        # reset termination criteria
        start_iterations = self.env.time
        admm_iter = 0

        # start the ADMM iteration loop
        while True:
            start_opt = time.time()

            # Solve local optimization
            result = self._solve_local_optimization(
                opt_inputs=opt_inputs,
                current_iteration=admm_iter,
                start_time=start_iterations,
            )

            # admm coordination step
            self.send_coupling_values(result)
            self._status = ModuleStatus.waiting_for_other_agents
            self._receive_variables(start=start_opt)
            self._status = ModuleStatus.updating
            self._set_mean_coupling_values()
            self.update_lambda()
            self.reset_participants_ready()

            # check termination
            admm_iter += 1
            if self._check_termination(admm_iter, start_iterations):
                break

        self.deregister_all_participants()
        self.set_actuation(result)

    def _solve_local_optimization(
        self,
        opt_inputs: Dict[str, AgentVariable],
        current_iteration: int,
        start_time: float,
    ) -> Results:
        """
        Performs the local optimization and returns the result.
        Args:
            opt_inputs: dict with AgentVariables that stay constant between
                optimizations
            current_iteration: current iteration number
            start_time: environment time at start of ADMM algorithm

        Returns:
            DataFrame of all optimization variables.
        """
        updated_couplings = self.collect_couplings_for_optimization()
        opt_inputs.update(updated_couplings)
        self.logger.info("Solving local optimization #%s.", current_iteration)
        self._status: ModuleStatus = ModuleStatus.optimizing
        result = self.optimization_backend.solve(start_time, opt_inputs)
        self.logger.info("Solved local optimization #%s.", current_iteration)
        return result

    def _perform_registration(self):
        """Registers participants in current round"""
        self._status: ModuleStatus = ModuleStatus.at_registration
        self.logger.info("Start registration of round at %s.", self.env.now)

        # shift initial values for multipliers and coupling outputs
        self._shift_and_send_coupling_outputs()
        self._shift_multipliers()

        # accept registrations within a fixed time (handled by callbacks)
        time.sleep(self.config.registration_period)
        self._status: ModuleStatus = ModuleStatus.updating
        self.logger.info("%s: Finished registration of round")

    def _check_termination(self, admm_iter: int, start_iteration: float) -> bool:
        """

        Args:
            admm_iter: current iteration number
            start_iteration: environment time at which current optimization
                began

        Returns:
            True, if the algorithm should be terminated,
            False, if it should continue
        """
        self.logger.debug("Finished iteration no. %s.", admm_iter)

        # check timeout
        available_runtime = self.config.time_step - self.config.registration_period
        if self.env.now - start_iteration > available_runtime:
            self.logger.warning(
                "ADMM did not converge within the specified sampling time "
                "of %ss. Terminating current control step.",
                self.config.time_step,
            )
            return True

        # check maximum iterations
        if admm_iter >= self.config.max_iterations:
            self.logger.warning(
                "ADMM did not converge within the maximum iteration number "
                "of %s. Terminating current control step.",
                self.config.max_iterations,
            )
            return True

        return False

    def _receive_variables(self, start):
        """Wait until all coupling variables arrive from the other systems."""

        timeout = self.config.iteration_timeout
        remaining_time = max(timeout - (time.time() - start), 0)
        for participant in self.all_coupling_statuses():
            if participant.status == ParticipantStatus.not_participating:
                continue
            try:
                var = participant.received.get(timeout=remaining_time)
                participant.variable = var
                participant.status = ParticipantStatus.confirmed
            except queue.Empty:
                participant.de_register()
                source = participant.variable.source
                coup = participant.variable.alias
                self.logger.info(
                    "De-registered participant %s from "
                    "coupling %s as it was too slow.",
                    source,
                    coup,
                )

            remaining_time = max(timeout - (time.time() - start), 0)

    def all_coupling_statuses(self) -> Iterable[ADMMParticipation]:
        """Gives and iterator of all ADMMParticipation that are registered."""
        for coup_participants in self.registered_participants.values():
            for participant in coup_participants.values():
                yield participant

    def _shift(self, sequence: List[float], grid: List[float]) -> List[float]:
        """
        Shifts the sequence forward by one sampling time.
        Args:
            sequence:   Sequence of variable values.
            grid:       Timestamps belonging to the sequence starting from 0.

        Returns:
            The shifted list with the last values duplicated.
        """
        # get index of first grid point greater self.ts
        index = next(x[0] for x in enumerate(grid) if x[1] >= self.config.time_step)
        shifted = sequence[index:] + sequence[-index:]
        return shifted

    def _shift_multipliers(self):
        """Shifts lagrange multipliers by one sampling interval. If a scalar
        is given, expands to the correct length."""
        for coup in self.cons_and_exchange:
            grid = self.optimization_backend.coupling_grid
            var = self._admm_variables[coup.multiplier]
            val = var.value
            if len(val) == 1:
                val = val * len(grid)
            val = self._shift(sequence=val, grid=grid)
            self._admm_variables[var.name].value = val

    def _shift_and_send_coupling_outputs(self):
        """Shifts global coupling variables by one sampling interval. If a
        scalar is given, expands to the correct length.
        Sets the values as output to the data_broker, sending them."""

        self.logger.info("Sending initial coupling outputs ...")
        for coupling in self.cons_and_exchange:
            grid = self.optimization_backend.coupling_grid
            length = len(grid)

            # shift output variable
            var = self._admm_variables[coupling.local]
            val = var.value

            # expand lists that were _finished_discretization with a scalar
            if len(val) == 1:
                val = val * length
            val = self._shift(sequence=val, grid=grid)

            self.send_coupling_variable(var.name, val)

    def assert_mpc_variables_are_in_model(self):
        unassigned_model = super().assert_mpc_variables_are_in_model()

        for coup in self.config.couplings + self.config.exchange:
            if coup.name in unassigned_model["inputs"]:
                unassigned_model["inputs"] = self.assert_subset(
                    [coup.name], unassigned_model["inputs"], "Couplings"
                )
            elif coup.name in unassigned_model["outputs"]:
                unassigned_model["outputs"] = self.assert_subset(
                    [coup.name], unassigned_model["outputs"], "Couplings"
                )
            elif coup.name in unassigned_model["states"]:
                unassigned_model["states"] = self.assert_subset(
                    [coup.name], unassigned_model["states"], "Couplings"
                )
        return unassigned_model

    @property
    def registered_participants(self) -> Dict[str, Dict[str, ADMMParticipation]]:
        """Dictionary containing all other agents this agent shares variables with.
        Ordered in a two-layer form, with variables at the first layer and
        agents at the second layer. Contains ADMMParticipation objects at
        the base layer.

        Examples:
            self.registered_participants =
            {'coupling_var_1': {'src_of_agent1': status_1,
                                'src_of_agent2': status_2,
                                'src_of_agent3': status_3}
             'coupling_var_1': {'src_of_agent3': status_a,
                                'src_of_agent2': status_b,
                                'src_of_agent4': status_c}
            }
            here, <status> refers to an ADMMParticipation object.
        """
        return self._registered_participants

    @registered_participants.setter
    def registered_participants(self, reg_par: Dict):
        self._registered_participants = reg_par

    @property
    def cons_and_exchange(self) -> List[Union[adt.ExchangeEntry, adt.CouplingEntry]]:
        return self.var_ref.exchange + self.var_ref.couplings

    def reset_participants_ready(self):
        """Sets the ready status of all participating agents to False."""
        for coup_participants in self.registered_participants.values():
            for participant in coup_participants.values():
                if participant.received.qsize():
                    participant.status = ParticipantStatus.available
                else:
                    participant.status = ParticipantStatus.not_available

    def deregister_all_participants(self):
        """Sets the participating status of all participating agents to
        False."""
        self.logger.info("De-registering all participants for next round.")
        for coup_participants in self.registered_participants.values():
            for participant in coup_participants.values():
                participant.de_register()

    def participant_callback(self, variable: AgentVariable):
        """Puts received variables in the correct queue, depending on
        registration status of this agent."""
        if variable.source.agent_id != self.agent.id:
            self.receive_participant(variable)

    def receive_participant(self, variable: AgentVariable):
        """Set the participation to true for the given coupling input."""
        # Create copy just in case
        reg_par_of_coupling = self.registered_participants[variable.alias].copy()

        # add variables that were seen the first time
        if variable.source not in reg_par_of_coupling:
            self.logger.info(
                "Initially registered variable '%s' from '%s'.",
                variable.alias,
                variable.source,
            )
            reg_par_of_coupling[variable.source] = ADMMParticipation(variable=variable)
        neighbor: ADMMParticipation = reg_par_of_coupling[variable.source]

        # perform registration at start of round
        if self._status == ModuleStatus.at_registration:
            self.logger.debug(
                "Registered variable '%s' from '%s' for this round.",
                variable.alias,
                variable.source,
            )
            neighbor.empty_memory()
            neighbor.status = ParticipantStatus.not_available
            neighbor.variable = variable

        # confirm new trajectory during admm iterations
        if self._status in (
            ModuleStatus.waiting_for_other_agents,
            ModuleStatus.optimizing,
            ModuleStatus.updating,
        ):
            try:
                neighbor.received.put_nowait(variable)
                neighbor.status = ParticipantStatus.available
                self.logger.debug(
                    "Received variable '%s' from '%s' and set to " "ready: 'True'.",
                    variable.alias,
                    variable.source,
                )
            except queue.Full:
                # status.de_register()
                source = neighbor.variable.source
                coup = neighbor.variable.alias
                self.logger.error(
                    "De-registered participant %s from coupling %s as it "
                    "sends messages too quickly.",
                    source,
                    coup,
                )
            if neighbor.received.qsize() > 2:
                self.logger.error(f"Queue is too full {neighbor.received.qsize()}")
            neighbor.variable = variable

        # Set the altered copy again
        self.registered_participants[variable.alias] = reg_par_of_coupling

    def get_participants_values(self, coupling_alias: str) -> List[pd.Series]:
        """Get the values of all agents for a coupling variable."""
        values = []
        for participant in self.registered_participants[coupling_alias].values():
            if participant.status == ParticipantStatus.confirmed:
                values.append(participant.variable.value)
        if not values:
            self.logger.warning("Did not get participants values for this round")
        return values

    def send_coupling_values(self, solution: Results):
        """
        Sets the coupling outputs to the data_broker, which automatically sends them.

        Args:
            solution: Output dictionary from optimization_backend.solve().
        """
        self.logger.info("Sending optimal values to other agents.")
        for coup in self.cons_and_exchange:
            self.send_coupling_variable(coup.local, list(solution[coup.name]))

    def _set_mean_coupling_values(self):
        """Computes the current global value of a coupling variable and saves
        it in the data_broker."""
        for coupling in self.var_ref.couplings:
            # Get own coupling variable version
            own_coup_var = self._admm_variables[coupling.local]
            own_coup_value = own_coup_var.value
            coup_alias = own_coup_var.alias

            # Get variables values:
            other_coup_values = self.get_participants_values(coup_alias)

            # Add own value
            other_coup_values.append(own_coup_value)

            # Build mean over all values
            other_coup_values = np.array(other_coup_values)
            mean_coup_value = list(np.mean(other_coup_values, axis=0))
            self._admm_variables[coupling.mean].value = mean_coup_value
            self.logger.debug(
                "Updated mean_%s = %s", own_coup_var.name, mean_coup_value
            )

        for exchange in self.var_ref.exchange:
            own_exchange_var = self._admm_variables[exchange.local]
            own_exchange_value = own_exchange_var.value
            exchange_alias = own_exchange_var.alias

            # Get variables values:
            other_coup_values = self.get_participants_values(exchange_alias)

            # Add own value
            other_coup_values.append(own_exchange_value)

            # Build mean over all values
            other_coup_values = np.array(other_coup_values)
            mean_coup_value = np.mean(other_coup_values, axis=0)
            mean_diff = list(own_exchange_value - mean_coup_value)

            self._admm_variables[exchange.mean_diff].value = mean_diff
            self.logger.debug(
                "Updated mean_%s = %s", own_exchange_var.name, mean_coup_value
            )

    def _solve_local_optimization_debug(
        self,
        opt_inputs: Dict[str, AgentVariable],
        current_iteration: int,
        start_time: float,
    ) -> pd.DataFrame:
        """
        USED FOR DEBUGGING, SKIPS CASADI
        Performs the local optimization and returns the result.
        Args:
            opt_inputs: dict with AgentVariables that stay constant between
                optimizations
            current_iteration: current iteration number
            start_time: environment time at start of ADMM algorithm

        Returns:
            DataFrame of all optimization variables.
        """
        updated_couplings = self.collect_couplings_for_optimization()
        opt_inputs.update(updated_couplings)
        self.logger.info("Solving local optimization #%s.", current_iteration)
        self._status: ModuleStatus = ModuleStatus.optimizing
        grid = self.optimization_backend.coupling_grid
        result = {}
        for coup in self.config.couplings + self.config.controls + self.config.states:
            result[coup.name] = [coup.value] * len(grid)
        result = pd.DataFrame(result)
        self.logger.info("Solved local optimization #%s.", current_iteration)
        self.logger.debug(
            "Coupling variable #%s.", list(result[self.config.couplings[0].name])
        )
        return result

    def send_coupling_variable(self, name: str, value: mpc_datamodels.MPCValue):
        """Sends an admm coupling variable through the data_broker and sets its
        value locally"""
        var = self._admm_variables[name]
        var.value = value
        self.agent.data_broker.send_variable(var)

    def update_lambda(self):
        """
        Performs the update of the lagrange multipliers.
        lambda^k+1 := lambda^k - rho*(z-x_i)
        """
        self.logger.info("Updating lambda variables for all couplings")
        for coupling in self.var_ref.couplings:
            # Get current lambda value:
            coup_name = coupling.name
            lambda_coupling = self._admm_variables[coupling.multiplier].value
            lambda_coupling = np.array(lambda_coupling)
            self.logger.debug("Updating lambda_%s = %s", coup_name, lambda_coupling)

            own_coup_value = self._admm_variables[coupling.local].value
            own_coup_value = np.array(own_coup_value)
            mean_coup_value = self._admm_variables[coupling.mean].value
            mean_coup_value = np.array(mean_coup_value)

            # Calc update
            updated_value = lambda_coupling - self.config.penalty_factor * (
                mean_coup_value - own_coup_value
            )
            updated_value = updated_value.tolist()
            # Set value to data_broker
            self._admm_variables[coupling.multiplier].value = updated_value
            self.logger.debug("Updated lambda_%s = %s", coupling.name, updated_value)

        for exchange in self.var_ref.exchange:
            # Get current lambda value:
            lambda_coupling = self._admm_variables[exchange.multiplier].value
            lambda_coupling = np.array(lambda_coupling)
            self.logger.debug("Updating lambda_%s = %s", exchange.name, lambda_coupling)

            own_coup_value = np.array(self._admm_variables[exchange.local].value)
            diff_coup_value = np.array(self._admm_variables[exchange.mean_diff].value)

            # Calc update
            updated_value = lambda_coupling - self.config.penalty_factor * (
                diff_coup_value - own_coup_value
            )
            updated_value = updated_value.tolist()
            # Set value to data_broker
            self._admm_variables[exchange.multiplier].value = updated_value
            self.logger.debug("Updated lambda_%s = %s", exchange.name, updated_value)

    def get_results(self) -> Optional[pd.DataFrame]:
        """Read the results that were saved from the optimization backend and
        returns them as Dataframe.

        Returns:
            (results, stats) tuple of Dataframes.
        """
        results_file = self.optimization_backend.config.results_file
        if results_file is None:
            self.logger.info("No results were saved .")
            return None
        try:
            results, stats = self.read_results_file(results_file)
            return results
        except FileNotFoundError:
            self.logger.error("ADMM results file %s was not found.", results_file)
            return None

    @property
    def penalty_factor_var(self) -> MPCVariable:
        return MPCVariable(name="penalty_factor", value=self.config.penalty_factor)

    def _setup_var_ref(self) -> adt.VariableReference:
        # Extend var_ref with coupling variables
        return adt.VariableReference.from_config(self.config)

    def _setup_optimization_backend(self) -> ADMMBackend:
        self._admm_variables = self._create_couplings()
        return super()._setup_optimization_backend()

    def _create_couplings(self) -> dict[str, MPCVariable]:
        """Map coupling variables based on already setup model"""
        # Check if coupling even exist

        # Map couplings:
        _couplings = []
        # and generate new variables for admm:
        _admm_variables: dict[str, MPCVariable] = {}
        for coupling in self.config.couplings:
            coupling.source = Source(agent_id=self.agent.id)
            coupling.shared = True
            _couplings.append(coupling)

            # Create two new variables for each coupling:
            # 1. lambda variables in both cases.
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

            # add callback to receive this value
            broker_funcs = [
                self.agent.data_broker.deregister_callback,
                self.agent.data_broker.register_callback,
            ]

            for broker_func in broker_funcs:
                broker_func(
                    alias=alias,
                    source=None,
                    callback=self.participant_callback,
                )
            self.registered_participants.update({alias: {}})

        # Exchange variables
        _exchange_vars = []
        # and generate new variables for admm:
        for exchange_var in self.config.exchange:
            exchange_var.source = Source(agent_id=self.agent.id)
            exchange_var.shared = True
            _exchange_vars.append(exchange_var)

            # Create two new variables for each coupling:
            # 1. lambda variables in both cases.
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

            # add callback to receive this value
            broker_funcs = [
                self.agent.data_broker.deregister_callback,
                self.agent.data_broker.register_callback,
            ]

            for broker_func in broker_funcs:
                broker_func(
                    alias=alias,
                    source=None,
                    callback=self.participant_callback,
                )
            self.registered_participants.update({alias: {}})
        return _admm_variables

    def collect_variables_for_optimization(
        self, var_ref: mpc_datamodels.VariableReference = None
    ) -> dict[str, AgentVariable]:
        """Gets all variables noted in the var ref and puts them in a flat
        dictionary."""
        if var_ref is None:
            var_ref = self.var_ref

        # config variables
        variables = {v: self.get(v) for v in var_ref.all_variables()}
        for coup_entry in var_ref.exchange + var_ref.couplings:
            lagged_admm_var = coup_entry.lagged
            original_name = coup_entry.name
            variable = self.get(original_name)
            if original_name in self.history:
                past_values = self.history[original_name]
                variable = MPCVariable(
                    name=lagged_admm_var, value=pd.Series(past_values)
                )
            variables[lagged_admm_var] = variable

        # history variables
        for hist_var in self._lags_dict_seconds:
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

        return {**variables, **self._internal_variables}


class LocalADMMConfig(ADMMConfig):
    sync_delay: float = 0.001
    registration_delay: float = 0.1


class LocalADMM(ADMM):
    config: LocalADMMConfig

    @property
    def sync_delay(self) -> float:
        """Timeout value used to sync local admm processes. Should be very
        small."""
        return self.config.sync_delay

    @property
    def registration_delay(self) -> float:
        """Timeout value used to wait one on registration. Waits in real time
        (time.sleep)"""
        return self.config.registration_delay

    def process(self):
        first_registration = True
        while True:
            start_round = self.env.time

            # Register participants in current round
            self.logger.info("Start registration of round at %s.", self.env.now)
            self._status = ModuleStatus.at_registration
            yield self.env.timeout(self.sync_delay)

            # shift initial values for multipliers and coupling outputs
            self._shift_and_send_coupling_outputs()
            self._shift_multipliers()
            self.pre_computation_hook()
            yield self.env.timeout(self.sync_delay)
            self._status = ModuleStatus.optimizing
            self.logger.info("Finished registration of round")
            yield self.env.timeout(self.sync_delay)

            if first_registration:
                time.sleep(self.registration_delay)
                first_registration = False

            # get optimization inputs
            self._set_mean_coupling_values()
            opt_inputs = self.collect_variables_for_optimization()
            # reset termination criteria
            start_iterations = self.env.time
            admm_iter = 0

            # start the ADMM iteration loop
            while True:
                # Solve local optimization
                start_opt = time.time()
                updated_couplings = self.collect_couplings_for_optimization()
                opt_inputs.update(updated_couplings)
                self.logger.info("Solving local optimization #%s.", admm_iter)
                self._status = ModuleStatus.optimizing
                result = self.optimization_backend.solve(start_iterations, opt_inputs)
                self.logger.info("Solved local optimization #%s.", admm_iter)

                # admm coordination step
                yield self.env.timeout(self.sync_delay)
                self.send_coupling_values(result)
                yield self.env.timeout(self.sync_delay)
                self._status = ModuleStatus.waiting_for_other_agents
                self._receive_variables(start=start_opt)
                yield self.env.timeout(self.sync_delay)
                self._status = ModuleStatus.updating
                self._set_mean_coupling_values()
                self.update_lambda()
                self.reset_participants_ready()
                yield self.env.timeout(self.sync_delay)

                # check termination
                admm_iter += 1
                if self._check_termination(admm_iter, start_iterations):
                    break

            self.deregister_all_participants()
            self.set_actuation(result)
            self._status = ModuleStatus.sleeping

            time_spent_on_sync_delay = self.env.time - start_round
            yield self.env.timeout(self.config.time_step - time_spent_on_sync_delay)
