"""
Defines classes that coordinate an ADMM process.
"""

import os
import time
from ast import literal_eval
from pathlib import Path
from typing import Dict, List, Optional
import queue
import logging
from dataclasses import asdict
import threading
import math

from pydantic import field_validator, Field
import numpy as np
import pandas as pd

from agentlib.core.agent import Agent
from agentlib.core.datamodels import AgentVariable, Source
from pydantic_core.core_schema import FieldValidationInfo

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
    wait_time_on_start_iters: float = Field(
        title="wait_on_start_iterations",
        default=0.1,
        description="wait_on_start_iterations",
    )
    registration_period: float = Field(
        title="registration_period",
        default=5,
        description="Time spent on registration before each optimization",
    )
    admm_iter_max: int = Field(
        title="admm_iter_max",
        default=20,
        description="Maximum number of ADMM iterations before termination of control "
        "step.",
    )
    time_step: float = Field(
        title="time_step",
        default=600,  # seconds
        description="Sampling interval of between two control steps. Will be used in "
        "the discretization for MPC.",
    )
    sampling_time: Optional[float] = Field(
        default=None,  # seconds
        description="Sampling interval for control steps. If None, will be the same as"
        " time step. Does not affect the discretization of the MPC, "
        "only the interval with which there will be optimization steps.",
        validate_default=True,
    )
    prediction_horizon: int = Field(
        title="prediction_horizon",
        default=10,
        description="Prediction horizon of participating agents.",
    )
    abs_tol: float = Field(
        title="abs_tol",
        default=1e-3,
        description="Absolute stopping criterion.",
    )
    rel_tol: float = Field(
        title="rel_tol",
        default=1e-3,
        description="Relative stopping criterion.",
    )
    primal_tol: float = Field(
        default=1e-3,
        description="Absolute primal stopping criterion.",
    )
    dual_tol: float = Field(
        default=1e-3,
        description="Absolute dual stopping criterion.",
    )
    use_relative_tolerances: bool = Field(
        default=True,
        description="If True, use abs_tol and rel_tol, if False us prim_tol and "
        "dual_tol.",
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
    save_solve_stats: bool = Field(
        default=False,
        description="When True, saves the solve stats to a file.",
    )
    solve_stats_file: str = Field(
        default="admm_stats.csv",  # seconds
        description="File name for the solve stats.",
    )
    save_iter_interval: int = Field(
        default=1000,
    )

    @field_validator("solve_stats_file")
    @classmethod
    def solve_stats_file_is_csv(cls, file: str):
        assert file.endswith(".csv")
        return file

    @field_validator("sampling_time")
    @classmethod
    def default_sampling_time(cls, samp_time, info: FieldValidationInfo):
        if samp_time is None:
            samp_time = info.data["time_step"]
        return samp_time


class ADMMCoordinator(Coordinator):
    config: ADMMCoordinatorConfig

    def __init__(self, *, config: dict, agent: Agent):
        if agent.env.config.rt:
            self.process = self._realtime_process
            self.registration_callback = self._real_time_registration_callback
        else:
            self.process = self._fast_process
            self.registration_callback = self._sequential_registration_callback

        super().__init__(config=config, agent=agent)
        self._coupling_variables: Dict[str, adt.ConsensusVariable] = {}
        self._exchange_variables: Dict[str, adt.ExchangeVariable] = {}
        self._agents_to_register = queue.Queue()
        self.agent_dict: Dict[str, adt.AgentDictEntry] = {}
        self._registration_queue: queue.Queue = queue.Queue()
        self._registration_lock: threading.Lock = threading.Lock()
        self.penalty_parameter = self.config.penalty_factor
        self._iteration_stats: pd.DataFrame = pd.DataFrame(
            columns=["primal_residual", "dual_residual"]
        )
        self._primal_residuals_tracker: List[float] = []
        self._dual_residuals_tracker: List[float] = []
        self._penalty_tracker: List[float] = []
        self._performance_tracker: List[float] = []
        self.start_algorithm_at: float = 0
        self._performance_counter: float = time.perf_counter()

    def _realtime_process(self):
        """Starts a thread to run next to the environment (to prevent a long blocking
        process). Periodically informs the thread of the next optimization."""
        self._start_algorithm = threading.Event()

        thread_proc = threading.Thread(
            target=self._realtime_process_thread,
            name=f"{self.source}_ProcessThread",
            daemon=True,
        )
        thread_proc.start()
        self.agent.register_thread(thread=thread_proc)

        thread_reg = threading.Thread(
            target=self._handle_registrations,
            name=f"{self.source}_RegistrationThread",
            daemon=True,
        )
        thread_reg.start()
        self.agent.register_thread(thread=thread_reg)

        while True:
            self._start_algorithm.set()
            yield self.env.timeout(self.config.sampling_time)

    def _realtime_process_thread(self):
        while True:
            self._status = cdt.CoordinatorStatus.sleeping
            self._start_algorithm.wait()
            self._start_algorithm.clear()
            with self._registration_lock:
                self._realtime_step()
            if self._start_algorithm.isSet():
                self.logger.error(
                    "%s: Start of ADMM round was requested before "
                    "last one finished. Skipping cycle."
                )
                self._start_algorithm.clear()

    def _realtime_step(self):
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

    def _wait_non_rt(self):
        """Returns a triggered event. Cedes control to the simpy event queue for a
        short moment. This is required in fast-as-possible simulations, to allow
        other agents to react via callbacks."""
        return self.env.timeout(0.001)

    def _fast_process(self):
        """Process function for use in fast-as-possible simulations. Regularly yields
        control back to the environment, to allow the callbacks to run."""
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
        """"""
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

    def _agents_with_status(self, status: cdt.AgentStatus) -> List[Source]:
        """Returns an iterator with all agents sources that are currently on
        this status."""
        active_agents = [s for (s, a) in self.agent_dict.items() if a.status == status]
        return active_agents

    def _check_convergence(self, iteration) -> bool:
        """
        Checks the convergence of the algorithm. Returns True if yes,
        False if no.
        Returns:
            Tuple of (converged, primal residual norm, dual residual norm)

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

        # primal_residual = np.concatenate(primal_residuals)
        # dual_residual = np.concatenate(dual_residuals)

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
            sqrt_p = math.sqrt(len(flat_multipliers))
            sqrt_n = math.sqrt(len(flat_locals))  # not actually n, but best we can do
            eps_pri = (
                sqrt_p * self.config.abs_tol + self.config.rel_tol * primal_scaling
            )
            eps_dual = sqrt_n * self.config.abs_tol + self.config.rel_tol * dual_scaling
            converged = prim_norm < eps_pri and dual_norm < eps_dual
        else:
            converged = (
                prim_norm < self.config.primal_tol and dual_norm < self.config.dual_tol
            )

        if converged:
            return True
        return False

    def _save_stats(self, iterations: int) -> None:
        """
        Args:
            iterations: Which iteration of the ADMM algorithm are we when this function
             is called?
        """
        section_length = len(self._penalty_tracker)
        section_start = iterations - section_length
        index = [
            (self.start_algorithm_at, i + section_start) for i in range(section_length)
        ]

        path = Path(self.config.solve_stats_file)
        header = not path.is_file()
        stats = pd.DataFrame(
            {
                "primal_residual": self._primal_residuals_tracker,
                "dual_residual": self._dual_residuals_tracker,
                "penalty_parameter": self._penalty_tracker,
                "wall_time": self._performance_tracker,
            },
            index=index,
        )
        self._penalty_tracker = []
        self._dual_residuals_tracker = []
        self._primal_residuals_tracker = []
        self._performance_tracker = []
        path.parent.mkdir(exist_ok=True, parents=True)
        stats.to_csv(path_or_buf=path, header=header, mode="a")

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
        Returns:

        """

        # create an iterator for all agents which are ready for this round
        active_agents: [str, adt.AgentDictEntry] = (
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

    def registration_callback(self, variable: AgentVariable):
        self.logger.debug(f"receiving {variable.name} from {variable.source}")
        if variable.source not in self.agent_dict:
            self.agent_dict[variable.source] = adt.AgentDictEntry(
                name=variable.source,
                status=cdt.AgentStatus.pending,
            )
            self._send_parameters_to_agent(variable)
            self.logger.info(
                f"Coordinator got request agent {variable.source} and set to "
                f"'pending'."
            )
            return
        # complete registration of pending agents
        if self.agent_dict[variable.source].status is cdt.AgentStatus.pending:
            self.register_agent(variable=variable)

    def _sequential_registration_callback(self, variable: AgentVariable):
        """Handles the registration for sequential i.e. local coordinators. Variables
        are handled immediately."""
        self.logger.debug(f"receiving {variable.name} from {variable.source}")
        self._initial_registration(variable)

    def _real_time_registration_callback(self, variable: AgentVariable):
        """Handles the registration for realtime coordinators. Variables are put in a
        queue and a thread registers them when it is safe to do so."""
        self.logger.debug(f"receiving {variable.name} from {variable.source}")
        self._registration_queue.put(variable)

    def _initial_registration(self, variable: AgentVariable):
        """Handles initial registration of a variable. If it is unknown, add it to
        the agent_dict and send it the global parameters. If it is sending its
        confirmation with initial trajectories,
        refer to the actual registration function."""
        if variable.source not in self.agent_dict:
            self.agent_dict[variable.source] = adt.AgentDictEntry(
                name=variable.source,
                status=cdt.AgentStatus.pending,
            )
            self._send_parameters_to_agent(variable)
            self.logger.info(
                f"Coordinator got request agent {variable.source} and set to "
                f"'pending'."
            )

        # complete registration of pending agents
        elif self.agent_dict[variable.source].status is cdt.AgentStatus.pending:
            self.register_agent(variable=variable)

    def _handle_registrations(self):
        """Performs registration tasks while the algorithm is on standby."""

        while True:
            # add new agent to dict and send them global parameters
            variable = self._registration_queue.get()

            with self._registration_lock:
                self._initial_registration(variable)

    def _wrap_up_algorithm(self, iterations):
        self._save_stats(iterations=iterations)
        self.penalty_parameter = self.config.penalty_factor

    def get_results(self) -> pd.DataFrame:
        """Reads the results on iteration data if they were saved."""
        results_file = self.config.solve_stats_file
        try:
            df = pd.read_csv(results_file, index_col=0, header=0)
            new_ind = [literal_eval(i) for i in df.index]
            df.index = pd.MultiIndex.from_tuples(new_ind)
            return df
        except FileNotFoundError:
            self.logger.error("Results file %s was not found.", results_file)
            return pd.DataFrame()

    def cleanup_results(self):
        results_file = self.config.solve_stats_file
        if not results_file:
            return
        os.remove(results_file)
