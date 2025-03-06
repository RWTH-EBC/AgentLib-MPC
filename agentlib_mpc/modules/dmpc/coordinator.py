import logging
import time
from dataclasses import asdict
from typing import Dict, List, Optional
import threading
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ast import literal_eval
from pydantic import Field, field_validator

from agentlib.core import (
    BaseModule,
    BaseModuleConfig,
    AgentVariable,
    Agent,
    Source,
    AgentVariables,
)
from agentlib_mpc.data_structures.coordinator_datatypes import (
    AgentStatus,
    RegistrationMessage,
)
import agentlib_mpc.data_structures.coordinator_datatypes as cdt


logger = logging.getLogger(__name__)


class CoordinatorConfig(BaseModuleConfig):
    """Base configuration for DMPC Coordinators."""

    maxIter: int = Field(default=10, description="Maximum number of iterations")
    time_out_non_responders: float = Field(
        default=1, description="Maximum wait time for subsystems in seconds"
    )
    wait_time_on_start_iters: float = Field(
        title="wait_on_start_iterations",
        default=0.1,
        description="Wait time after sending start iteration signal",
    )
    registration_period: float = Field(
        title="registration_period",
        default=5,
        description="Time spent on registration before each optimization",
    )
    time_step: float = Field(
        title="time_step",
        default=600,  # seconds
        description="Sampling interval between two control steps. Used in MPC discretization.",
    )
    sampling_time: Optional[float] = Field(
        default=None,  # seconds
        description="Sampling interval for control steps. If None, will be the same as time step.",
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
        description="If True, use abs_tol and rel_tol, if False use prim_tol and dual_tol.",
    )
    save_solve_stats: bool = Field(
        default=False,
        description="When True, saves the solve stats to a file.",
    )
    solve_stats_file: str = Field(
        default="dmpc_stats.csv",
        description="File name for the solve stats.",
    )
    save_iter_interval: int = Field(
        default=1000,
        description="Interval for saving iteration statistics",
    )
    messages_in: AgentVariables = [
        AgentVariable(name=cdt.REGISTRATION_A2C),
        AgentVariable(name=cdt.START_ITERATION_A2C),
        AgentVariable(name=cdt.OPTIMIZATION_A2C),
    ]
    messages_out: AgentVariables = [
        AgentVariable(name=cdt.REGISTRATION_C2A),
        AgentVariable(name=cdt.START_ITERATION_C2A),
        AgentVariable(name=cdt.OPTIMIZATION_C2A),
    ]
    shared_variable_fields: list[str] = ["messages_out"]

    @field_validator("sampling_time")
    @classmethod
    def default_sampling_time(cls, samp_time, info):
        if samp_time is None:
            samp_time = info.data["time_step"]
        return samp_time

    @field_validator("solve_stats_file")
    @classmethod
    def solve_stats_file_is_csv(cls, file: str):
        assert file.endswith(".csv")
        return file


class Coordinator(BaseModule):
    """Base class implementing coordination for distributed MPC"""

    config: CoordinatorConfig

    def __init__(self, *, config: dict, agent: Agent):
        # Determine process method based on environment configuration
        if agent.env.config.rt:
            self.process = self._realtime_process
            self.registration_callback = self._real_time_registration_callback
        else:
            self.process = self._fast_process
            self.registration_callback = self._sequential_registration_callback
        super().__init__(config=config, agent=agent)
        self.agent_dict: Dict[Source, cdt.AgentDictEntry] = {}
        self.status: cdt.CoordinatorStatus = cdt.CoordinatorStatus.sleeping
        self.received_variable = threading.Event()
        self._primal_residuals_tracker: List[float] = []
        self._dual_residuals_tracker: List[float] = []
        self._performance_tracker: List[float] = []
        self.start_algorithm_at: float = 0
        self._performance_counter: float = time.perf_counter()
        self._registration_queue = None
        self._registration_lock = threading.Lock()
        self._start_algorithm = None

    def _realtime_process(self):
        """Start threads to run alongside the environment for realtime operation."""
        self._start_algorithm = threading.Event()

        thread_proc = threading.Thread(
            target=self._realtime_process_thread,
            name=f"{self.source}_ProcessThread",
            daemon=True,
        )
        thread_proc.start()
        self.agent.register_thread(thread=thread_proc)

        # Initialize registration queue and start registration thread
        self._registration_queue = self._registration_queue or threading.Queue()
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
        """Thread for executing the optimization algorithm in realtime."""
        while True:
            self.status = cdt.CoordinatorStatus.sleeping
            self._start_algorithm.wait()
            self._start_algorithm.clear()
            with self._registration_lock:
                self._realtime_step()
            if self._start_algorithm.isSet():
                self.logger.error(
                    "%s: Start of optimization round was requested before "
                    "last one finished. Skipping cycle."
                )
                self._start_algorithm.clear()

    def _realtime_step(self):
        """Implement one step of the optimization algorithm in realtime mode."""
        # This method should be implemented by derived classes
        pass

    def _wait_non_rt(self):
        """Returns a triggered event to cede control to simpy event queue briefly."""
        return self.env.timeout(0.001)

    def _fast_process(self):
        """Process function for fast-as-possible simulations."""
        yield self._wait_non_rt()

        while True:
            # ------------------
            # start iteration
            # ------------------
            self.status = cdt.CoordinatorStatus.init_iterations
            self.start_algorithm_at = self.env.time
            self._performance_counter = time.perf_counter()
            self.set(cdt.START_ITERATION_C2A, True)
            # check for all_finished here
            yield self._wait_non_rt()
            if not list(self._agents_with_status(status=cdt.AgentStatus.ready)):
                self.logger.info(f"No Agents available at time {self.env.now}.")
                communication_time = self.env.time - self.start_algorithm_at
                yield self.env.timeout(self.config.sampling_time - communication_time)
                continue  # if no agents registered return early

            # Perform algorithm-specific iterations and convergence check
            # This will be implemented by derived classes

            self.status = cdt.CoordinatorStatus.sleeping
            time_spent_on_communication = self.env.time - self.start_algorithm_at
            yield self.env.timeout(
                self.config.sampling_time - time_spent_on_communication
            )

    def process(self):
        yield self.env.timeout(0.01)

        while True:
            # ------------------
            # start iteration
            # ------------------
            self.status = cdt.CoordinatorStatus.init_iterations
            # maybe this will hold information instead of "True"
            self.set(cdt.START_ITERATION_C2A, True)
            # check for all_finished here
            time.sleep(1)
            # ------------------
            # iteration loop
            # ------------------
            for iI in range(self.config.maxIter):
                # ------------------
                # optimization
                # ------------------
                # send
                self.status = cdt.CoordinatorStatus.optimization
                # set all agents to busy
                self.trigger_optimizations()

                # check for all finished here
                self._wait_for_ready()

                # receive
                ...
                # ------------------
                # perform update steps
                # ------------------
                self.status = cdt.CoordinatorStatus.updating
                ...
                # ------------------
                # check convergence
                # ------------------
                ...

            yield self.env.timeout(1)

    def trigger_optimizations(self):
        """
        Triggers the optimization for all agents with status ready.
        Returns:

        """
        send = self.agent.data_broker.send_variable
        for source, agent in self.agent_dict.items():
            if agent.status == cdt.AgentStatus.ready:
                value = agent.optimization_data.to_dict()
                self.logger.debug("Sending to %s with source %s", agent.name, source)
                self.logger.debug("Set %s to busy.", agent.name)
                agent.status = cdt.AgentStatus.busy
                message = AgentVariable(
                    name=cdt.OPTIMIZATION_C2A,
                    source=source,
                    value=value,
                )
                send(message)

    def register_callbacks(self):
        self.agent.data_broker.register_callback(
            alias=cdt.REGISTRATION_A2C,
            source=None,
            callback=self.registration_callback,
        )
        self.agent.data_broker.register_callback(
            alias=cdt.START_ITERATION_A2C,
            source=None,
            callback=self.init_iteration_callback,
        )
        self.agent.data_broker.register_callback(
            alias=cdt.OPTIMIZATION_A2C,
            source=None,
            callback=self.optim_results_callback,
        )

    def _real_time_registration_callback(self, variable: AgentVariable):
        """Handles the registration for realtime coordinators."""
        self.logger.debug(f"receiving {variable.name} from {variable.source}")
        if self._registration_queue is None:
            self._registration_queue = threading.Queue()
        self._registration_queue.put(variable)

    def _sequential_registration_callback(self, variable: AgentVariable):
        """Handles the registration for sequential coordinators."""
        self.logger.debug(f"receiving {variable.name} from {variable.source}")
        self._initial_registration(variable)

    def _initial_registration(self, variable: AgentVariable):
        """Handles initial registration of an agent."""
        if not (variable.source in self.agent_dict):
            entry = cdt.AgentDictEntry(
                name=variable.source,
                status=AgentStatus.pending,
            )
            self.agent_dict[variable.source] = entry
            self._send_parameters_to_agent(variable)
            self.logger.info(
                f"Coordinator got request from agent {variable.source} and set to 'pending'."
            )
        elif self.agent_dict[variable.source].status is cdt.AgentStatus.pending:
            self.register_agent(variable=variable)

    def _send_parameters_to_agent(self, variable: AgentVariable):
        """Sends parameters to an agent after registration request."""
        # To be implemented by derived classes
        pass

    def register_agent(self, variable: AgentVariable):
        """Registers an agent after it sends initial data."""
        # To be implemented by derived classes
        pass

    def optim_results_callback(self, variable: AgentVariable):
        """
        Saves the results of a local optimization.
        Args:
            variable:

        Returns:

        """
        entry = self.agent_dict[variable.source]
        entry.optimization_data = cdt.OptimizationData.from_dict(variable.value)
        self.agent_dict[variable.source].status = cdt.AgentStatus.ready
        self.received_variable.set()

    def init_iteration_callback(self, variable: AgentVariable):
        """
        Processes and Agents InitIteration confirmation.
        Args:
            variable:

        Returns:

        """
        if not self.status == cdt.CoordinatorStatus.init_iterations:
            # maybe set AgentStatus to something meaningful
            self.logger.error("Agent did not respond in time!")
            return

        if variable.value is not True:
            # did not receive acknowledgement
            return

        try:
            ag_dict_entry = self.agent_dict[variable.source]
        except KeyError:
            # likely did not finish registration of an agent yet, but the agent
            # already has its end registered and responds to the init_iterations.
            # Let it wait one round.
            return

        self.logger.debug(
            "Received 'StartIteration' confirmation from %s", variable.source
        )
        if ag_dict_entry.status != cdt.AgentStatus.standby:
            # if the status is not standby, the agent might still be in registration
            # phase, or something else occurred
            return
        ag_dict_entry.status = cdt.AgentStatus.ready
        self.received_variable.set()

    def _confirm_init_iteration(self, variable: AgentVariable):
        """Common functionality for checking init iteration callback."""
        if not self.status == cdt.CoordinatorStatus.init_iterations:
            self.logger.error("Agent did not respond in time!")
            return None

        if variable.value is False:
            return None

        try:
            ag_dict_entry = self.agent_dict[variable.source]
        except KeyError:
            # likely did not finish registration of an agent yet, but the agent
            # already has its end registered and responds to the init_iterations.
            # Let it wait one round.
            return None

        self.logger.debug(
            "Received 'StartIteration' confirmation from %s", variable.source
        )
        if ag_dict_entry.status != cdt.AgentStatus.standby:
            return None

        return ag_dict_entry

    @property
    def all_finished(self):
        """
        Returns:
            True if there are no busy agents, else False
        """
        for src, ag_entry in self.agent_dict.items():
            if ag_entry.status is cdt.AgentStatus.busy:
                return False
        return True

    def _agents_with_status(self, status: cdt.AgentStatus) -> List[Source]:
        """Returns a list of agent sources with the specified status."""
        return [s for (s, a) in self.agent_dict.items() if a.status == status]

    def _wait_for_ready(self):
        """Wait until all coupling variables arrive from the other systems."""
        self.received_variable.clear()
        self.logger.info("Start waiting for agents to finish computation.")
        while True:
            # check exit conditions
            if self.all_finished:
                count = 0
                for ag in self.agent_dict.values():
                    if ag.status == cdt.AgentStatus.ready:
                        count += 1
                self.logger.info("Got variables from all (%s) agents.", count)
                break

            # wait until a new item is put in the queue
            if self.received_variable.wait(timeout=self.config.time_out_non_responders):
                self.received_variable.clear()
            else:
                self._deregister_slow_participants()
                break

    def _deregister_slow_participants(self):
        """Sets all agents that are still busy to standby, so they won't be
        waited on again."""
        for agent in self.agent_dict.values():
            if agent.status == cdt.AgentStatus.busy:
                agent.status = cdt.AgentStatus.standby
                self.logger.info(
                    "De-registered agent %s as it was too slow.", agent.name
                )

    def _handle_registrations(self):
        """Thread for processing registration requests in realtime mode."""
        while True:
            # add new agent to dict and send them global parameters
            variable = self._registration_queue.get()

            with self._registration_lock:
                self._initial_registration(variable)

    def _save_stats(self, iterations: int, data_dict: dict) -> None:
        """Save iteration statistics to a file.

        Args:
            iterations: Current iteration number
            data_dict: Dictionary of data to save
        """
        if not self.config.save_solve_stats:
            return

        section_length = len(self._primal_residuals_tracker)
        if section_length == 0:
            return

        section_start = iterations - section_length
        index = [
            (self.start_algorithm_at, i + section_start) for i in range(section_length)
        ]

        path = Path(self.config.solve_stats_file)
        header = not path.is_file()
        stats = pd.DataFrame(data_dict, index=index)

        # Reset trackers
        self._primal_residuals_tracker = []
        self._dual_residuals_tracker = []
        self._performance_tracker = []

        path.parent.mkdir(exist_ok=True, parents=True)
        stats.to_csv(path_or_buf=path, header=header, mode="a")

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
        """Delete the results file."""
        results_file = self.config.solve_stats_file
        if not results_file:
            return
        try:
            os.remove(results_file)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    pass
