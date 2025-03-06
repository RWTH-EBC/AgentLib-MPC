"""
Defines classes that coordinate an ADMM process.
"""

import logging
import time
from typing import Dict, List, Optional
import queue
import threading
from dataclasses import asdict

from pydantic import Field
import numpy as np

from agentlib.core.agent import Agent
from agentlib.core.datamodels import AgentVariable, Source

from agentlib_mpc.data_structures import coordinator_datatypes as cdt
from agentlib_mpc.modules.dmpc.coordinator import Coordinator, CoordinatorConfig
import agentlib_mpc.data_structures.admm_datatypes as adt

logger = logging.getLogger(__name__)


class ADMMCoordinatorConfig(CoordinatorConfig):
    """Hold the config for ADMMCoordinator"""

    penalty_factor: float = Field(
        title="penalty_factor",
        default=10,
        description="Penalty factor of the ADMM algorithm. Should be equal "
        "for all agents.",
    )
    admm_iter_max: int = Field(
        title="admm_iter_max",
        default=20,
        description="Maximum number of ADMM iterations before termination of control "
        "step.",
    )
    penalty_change_threshold: float = Field(
        default=-1,
        description="When the primal residual is x times higher, vary the penalty "
        "parameter and vice versa.",
    )
    penalty_change_factor: float = Field(
        default=2,  # seconds
        description="Factor to vary the penalty parameter with.",
    )
    solve_stats_file: str = Field(
        default="admm_stats.csv",  # seconds
        description="File name for the solve stats.",
    )


class ADMMCoordinator(Coordinator):
    """Coordinator implementation for ADMM algorithm"""

    config: ADMMCoordinatorConfig

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        self._coupling_variables: Dict[str, adt.ConsensusVariable] = {}
        self._exchange_variables: Dict[str, adt.ExchangeVariable] = {}
        self._agents_to_register = queue.Queue()
        self.agent_dict: Dict[str, adt.AgentDictEntry] = {}
        self.penalty_parameter = self.config.penalty_factor
        self._penalty_tracker: List[float] = []

    def _initial_registration(self, variable: AgentVariable):
        """Handles initial registration of an agent with ADMM-specific entry type."""
        if not (variable.source in self.agent_dict):
            entry = adt.AgentDictEntry(  # Use ADMM-specific type
                name=variable.source,
                status=cdt.AgentStatus.pending,
            )
            self.agent_dict[variable.source] = entry
            self._send_parameters_to_agent(variable)
            self.logger.info(
                f"Coordinator got request from agent {variable.source} and set to 'pending'."
            )
        elif self.agent_dict[variable.source].status is cdt.AgentStatus.pending:
            self.register_agent(variable=variable)

    def _realtime_step(self):
        """Implement one step of the ADMM algorithm in realtime mode."""
        # ------------------
        # start iteration
        # ------------------
        self.status = cdt.CoordinatorStatus.init_iterations
        self.start_algorithm_at = self.env.time
        self._performance_counter = time.perf_counter()
        # maybe this will hold information instead of "True"
        self.set(cdt.START_ITERATION_C2A, True)
        # check for all_finished here
        time.sleep(self.config.wait_time_on_start_iters)
        if not list(self._agents_with_status(status=cdt.AgentStatus.ready)):
            self.logger.info(f"No Agents available at time {self.env.now}.")
            return  # if no agents registered return early
        self._update_mean_coupling_variables()
        self._shift_coupling_variables()
        # ------------------
        # iteration loop
        # ------------------
        admm_iter = 0
        for admm_iter in range(1, self.config.admm_iter_max + 1):
            # ------------------
            # optimization
            # ------------------
            # send
            self.status = cdt.CoordinatorStatus.optimization
            # set all agents to busy
            self.trigger_optimizations()

            # check for all finished here
            self._wait_for_ready()

            # ------------------
            # perform update steps
            # ------------------
            self.status = cdt.CoordinatorStatus.updating
            self._update_mean_coupling_variables()
            self._update_multipliers()
            # ------------------
            # check convergence
            # ------------------
            converged = self._check_convergence(admm_iter)
            if converged:
                self.logger.info("Converged within %s iterations. ", admm_iter)
                break
        else:
            self.logger.warning(
                "Did not converge within the maximum number of iterations " "%s. ",
                self.config.admm_iter_max,
            )
        self._wrap_up_algorithm(iterations=admm_iter)
        self.set(cdt.START_ITERATION_C2A, False)  # this signals the finish

    def _fast_process(self):
        """Process function for use in fast-as-possible simulations."""
        yield self._wait_non_rt()

        while True:
            # ------------------
            # start iteration
            # ------------------
            self.status = cdt.CoordinatorStatus.init_iterations
            self.start_algorithm_at = self.env.time
            self._performance_counter = time.perf_counter()
            self.set(cdt.START_ITERATION_C2A, True)
            yield self._wait_non_rt()
            if not list(self._agents_with_status(status=cdt.AgentStatus.ready)):
                self.logger.info(f"No Agents available at time {self.env.now}.")
                communication_time = self.env.time - self.start_algorithm_at
                yield self.env.timeout(self.config.sampling_time - communication_time)
                continue  # if no agents registered return early
            self._update_mean_coupling_variables()
            self._shift_coupling_variables()
            # ------------------
            # iteration loop
            # ------------------
            admm_iter = 0
            for admm_iter in range(1, self.config.admm_iter_max + 1):
                # ------------------
                # optimization
                # ------------------
                # send
                self.status = cdt.CoordinatorStatus.optimization
                # set all agents to busy
                self.trigger_optimizations()
                yield self._wait_non_rt()

                # check for all finished here
                self._wait_for_ready()

                # ------------------
                # perform update steps
                # ------------------
                self.status = cdt.CoordinatorStatus.updating
                self._update_mean_coupling_variables()
                self._update_multipliers()
                # ------------------
                # check convergence
                # ------------------
                converged = self._check_convergence(admm_iter)
                if converged:
                    self.logger.info("Converged within %s iterations. ", admm_iter)
                    break
            else:
                self.logger.warning(
                    "Did not converge within the maximum number of iterations " "%s. ",
                    self.config.admm_iter_max,
                )
            self._wrap_up_algorithm(iterations=admm_iter)
            self.set(cdt.START_ITERATION_C2A, False)  # this signals the finish
            self.status = cdt.CoordinatorStatus.sleeping
            time_spent_on_communication = self.env.time - self.start_algorithm_at
            yield self.env.timeout(
                self.config.sampling_time - time_spent_on_communication
            )

    def _update_mean_coupling_variables(self):
        """Calculates a new mean of the coupling variables."""
        active_agents = self._agents_with_status(cdt.AgentStatus.ready)
        for variable in self._coupling_variables.values():
            variable.update_mean_trajectory(sources=active_agents)
        for variable in self._exchange_variables.values():
            variable.update_diff_trajectories(sources=active_agents)

    def _shift_coupling_variables(self):
        """Shift coupling variables for the next time step."""
        for variable in self._coupling_variables.values():
            variable.shift_values_by_one(horizon=self.config.prediction_horizon)
        for variable in self._exchange_variables.values():
            variable.shift_values_by_one(horizon=self.config.prediction_horizon)

    def _update_multipliers(self):
        """Performs the multiplier update for the coupling variables."""
        rho = self.penalty_parameter
        active_agents = self._agents_with_status(cdt.AgentStatus.ready)
        for variable in self._coupling_variables.values():
            variable.update_multipliers(rho=rho, sources=active_agents)
        for variable in self._exchange_variables.values():
            variable.update_multiplier(rho=rho)

    def _check_convergence(self, iteration) -> bool:
        """
        Checks the convergence of the algorithm.

        Returns:
            bool: True if converged, False otherwise
        """
        primal_residuals = []
        dual_residuals = []
        active_agents = self._agents_with_status(cdt.AgentStatus.ready)
        flat_locals = []
        flat_means = []
        flat_multipliers = []

        for var in self._coupling_variables.values():
            prim, dual = var.get_residual(rho=self.penalty_parameter)
            primal_residuals.extend(prim)
            dual_residuals.extend(dual)
            locs = var.flat_locals(sources=active_agents)
            muls = var.flat_multipliers(active_agents)
            flat_locals.extend(locs)
            flat_multipliers.extend(muls)
            flat_means.extend(var.mean_trajectory)

        for var in self._exchange_variables.values():
            prim, dual = var.get_residual(rho=self.penalty_parameter)
            primal_residuals.extend(prim)
            dual_residuals.extend(dual)
            locs = var.flat_locals(sources=active_agents)
            muls = var.multiplier
            flat_locals.extend(locs)
            flat_multipliers.extend(muls)
            flat_means.extend(var.mean_trajectory)

        # compute residuals
        prim_norm = np.linalg.norm(primal_residuals)
        dual_norm = np.linalg.norm(dual_residuals)

        self._vary_penalty_parameter(primal_residual=prim_norm, dual_residual=dual_norm)
        self._penalty_tracker.append(self.penalty_parameter)
        self._primal_residuals_tracker.append(prim_norm)
        self._dual_residuals_tracker.append(dual_norm)
        self._performance_tracker.append(
            time.perf_counter() - self._performance_counter
        )

        self.logger.debug(
            "Finished iteration %s . \n Primal residual: %s \n Dual residual: " "%s",
            iteration,
            prim_norm,
            dual_norm,
        )
        if iteration % self.config.save_iter_interval == 0:
            self._save_stats(iterations=iteration)

        if self.config.use_relative_tolerances:
            # scaling factors for relative criterion
            primal_scaling = max(
                np.linalg.norm(flat_locals),
                np.linalg.norm(flat_means),  # Ax  # Bz
            )
            dual_scaling = np.linalg.norm(flat_multipliers)
            # compute tolerances for this iteration
            sqrt_p = np.sqrt(len(flat_multipliers))
            sqrt_n = np.sqrt(len(flat_locals))  # not actually n, but best we can do
            eps_pri = (
                sqrt_p * self.config.abs_tol + self.config.rel_tol * primal_scaling
            )
            eps_dual = sqrt_n * self.config.abs_tol + self.config.rel_tol * dual_scaling
            converged = prim_norm < eps_pri and dual_norm < eps_dual
        else:
            converged = (
                prim_norm < self.config.primal_tol and dual_norm < self.config.dual_tol
            )

        return converged

    def _save_stats(self, iterations: int) -> None:
        """Save iteration statistics to a file."""
        data_dict = {
            "primal_residual": self._primal_residuals_tracker,
            "dual_residual": self._dual_residuals_tracker,
            "penalty_parameter": self._penalty_tracker,
            "wall_time": self._performance_tracker,
        }
        super()._save_stats(iterations=iterations, data_dict=data_dict)
        self._penalty_tracker = []

    def _vary_penalty_parameter(self, primal_residual: float, dual_residual: float):
        """Determines a new value for the penalty parameter based on residuals."""
        mu = self.config.penalty_change_threshold
        tau = self.config.penalty_change_factor

        if mu <= 1:
            # do not perform varying penalty method if the threshold is set below 1
            return

        if primal_residual > mu * dual_residual:
            self.penalty_parameter = self.penalty_parameter * tau
        elif dual_residual > mu * primal_residual:
            self.penalty_parameter = self.penalty_parameter / tau

    def trigger_optimizations(self):
        """
        Triggers the optimization for all agents with status ready.
        """
        # create an iterator for all agents which are ready for this round
        active_agents = (
            (s, a)
            for (s, a) in self.agent_dict.items()
            if a.status == cdt.AgentStatus.ready
        )

        # aggregate and send trajectories per agent
        for source, agent in active_agents:
            # collect mean and multiplier per coupling variable
            mean_trajectories = {}
            multipliers = {}
            for alias in agent.coup_vars:
                coup_var = self._coupling_variables[alias]
                mean_trajectories[alias] = coup_var.mean_trajectory
                multipliers[alias] = coup_var.multipliers[source]
            diff_trajectories = {}
            multiplier = {}
            for alias in agent.exchange_vars:
                coup_var = self._exchange_variables[alias]
                diff_trajectories[alias] = coup_var.diff_trajectories[source]
                multiplier[alias] = coup_var.multiplier

            # package all coupling inputs needed for an agent
            coordi_to_agent = adt.CoordinatorToAgent(
                mean_trajectory=mean_trajectories,
                multiplier=multipliers,
                exchange_multiplier=multiplier,
                mean_diff_trajectory=diff_trajectories,
                target=source.agent_id,
                penalty_parameter=self.penalty_parameter,
            )

            self.logger.debug("Sending to %s with source %s", agent.name, source)
            self.logger.debug("Set %s to busy.", agent.name)

            # send values
            agent.status = cdt.AgentStatus.busy
            self.set(cdt.OPTIMIZATION_C2A, coordi_to_agent.to_json())

    def _send_parameters_to_agent(self, variable: AgentVariable):
        """Sends an agent the global parameters after a signup request."""
        admm_parameters = adt.ADMMParameters(
            prediction_horizon=self.config.prediction_horizon,
            time_step=self.config.time_step,
            penalty_factor=self.config.penalty_factor,
        )

        message = cdt.RegistrationMessage(
            agent_id=variable.source.agent_id, opts=asdict(admm_parameters)
        )
        self.set(cdt.REGISTRATION_C2A, asdict(message))

    def register_agent(self, variable: AgentVariable):
        """Registers the agent, after it sent its initial guess with correct
        vector length."""
        value = adt.AgentToCoordinator.from_json(variable.value)
        src = variable.source
        ag_dict_entry = self.agent_dict[variable.source]

        # loop over coupling variables of this agent
        for alias, traj in value.local_trajectory.items():
            coup_var = self._coupling_variables.setdefault(
                alias, adt.ConsensusVariable()
            )

            # initialize Lagrange-Multipliers and local solution
            coup_var.multipliers[src] = [0] * len(traj)
            coup_var.local_trajectories[src] = traj
            ag_dict_entry.coup_vars.append(alias)

        # loop over coupling variables of this agent
        for alias, traj in value.local_exchange_trajectory.items():
            coup_var = self._exchange_variables.setdefault(
                alias, adt.ExchangeVariable()
            )

            # initialize Lagrange-Multipliers and local solution
            coup_var.multiplier = [0] * len(traj)
            coup_var.local_trajectories[src] = traj
            ag_dict_entry.exchange_vars.append(alias)

        # set agent from pending to standby
        ag_dict_entry.status = cdt.AgentStatus.standby
        self.logger.info(
            f"Coordinator successfully registered agent {variable.source}."
        )

    def optim_results_callback(self, variable: AgentVariable):
        """
        Saves the results of a local optimization.
        Args:
            variable:

        Returns:

        """
        local_result = adt.AgentToCoordinator.from_json(variable.value)
        source = variable.source
        for alias, trajectory in local_result.local_trajectory.items():
            coup_var = self._coupling_variables[alias]
            coup_var.local_trajectories[source] = trajectory
        for alias, trajectory in local_result.local_exchange_trajectory.items():
            coup_var = self._exchange_variables[alias]
            coup_var.local_trajectories[source] = trajectory

        self.agent_dict[variable.source].status = cdt.AgentStatus.ready
        self.received_variable.set()

    def _wrap_up_algorithm(self, iterations):
        """Clean up at the end of an algorithm execution."""
        self._save_stats(iterations=iterations)
        self.penalty_parameter = self.config.penalty_factor
