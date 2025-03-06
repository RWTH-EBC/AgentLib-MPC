"""Module implementing base class for coordinated optimization agents."""

import logging
import abc
from dataclasses import asdict
from typing import Dict, List, Optional
import pandas as pd

from pydantic import Field

from agentlib.core import (
    BaseModule,
    BaseModuleConfig,
    AgentVariable,
    Agent,
    AgentVariables,
)
from agentlib.core.datamodels import Source
from agentlib_mpc.data_structures.coordinator_datatypes import RegistrationMessage
import agentlib_mpc.data_structures.coordinator_datatypes as cdt
from agentlib_mpc.modules.mpc_full import MPCConfig, MPC

logger = logging.getLogger(__name__)


class OptimizingAgentConfig(MPCConfig):
    """Base configuration for optimization agents coordinated by a coordinator."""

    request_frequency: float = Field(
        default=1, description="Wait time between signup_requests"
    )
    coordinator: Source = Field(description="Define the agents coordinator")
    messages_in: AgentVariables = [
        AgentVariable(name=cdt.REGISTRATION_C2A),
        AgentVariable(name=cdt.START_ITERATION_C2A),
        AgentVariable(name=cdt.OPTIMIZATION_C2A),
    ]
    messages_out: AgentVariables = [
        AgentVariable(name=cdt.REGISTRATION_A2C),
        AgentVariable(name=cdt.START_ITERATION_A2C),
        AgentVariable(name=cdt.OPTIMIZATION_A2C),
    ]
    registration_interval: float = Field(
        default=10,
        ge=0,
        description="Interval in seconds after which a registration attempt is made.",
    )
    shared_variable_fields: list[str] = ["messages_out"]

    # Common optimization parameters
    penalty_factor: float = 10
    prediction_horizon: int = 10
    time_step: float = 600
    couplings: List[AgentVariable] = []
    exchange: List[AgentVariable] = []


class OptimizingAgent(MPC):
    """Base class for optimization agents coordinated by a coordinator."""

    config: OptimizingAgentConfig

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        self._registered_coordinator: Source = None
        self._start_optimization_at: float = 0
        self._optimization_inputs: Dict[str, AgentVariable] = {}
        self._result: Optional[pd.DataFrame] = None

    def process(self):
        # send registration request to coordinator
        timeout = self.config.registration_interval
        while True:
            if not self._registered_coordinator:
                self.set(cdt.REGISTRATION_A2C, True)
            yield self.env.timeout(timeout)

    def register_callbacks(self):
        # callback used for registration process
        coordinator_agent = Source(agent_id=self.config.coordinator.agent_id)
        self.agent.data_broker.register_callback(
            alias=cdt.REGISTRATION_C2A,
            source=coordinator_agent,
            callback=self.registration_callback,
        )
        #
        # call back for iteration start (
        self.agent.data_broker.register_callback(
            alias=cdt.START_ITERATION_C2A,
            source=coordinator_agent,
            callback=self.init_iteration_callback,
        )
        #
        # call back for optimization (
        self.agent.data_broker.register_callback(
            alias=cdt.OPTIMIZATION_C2A,
            source=coordinator_agent,
            callback=self.optimize,
        )

    def pre_computation_hook(self):
        """
        This method is called in every computation step before the optimization starts.
        Overwrite this method in a derived subclass if you want to take some actions each time before the optimal control problem is solved.
        """
        pass

    def init_iteration_callback(self, variable: AgentVariable):
        """
        Callback that processes the coordinators 'startIteration' flag.
        Args:
            variable:

        """
        # value is True on start
        if variable.value:
            self._start_optimization_at = self.env.time
            # new measurement
            self.get_new_measurement()
            # shift trajectories
            self.shift_trajectories()
            # custom function which can be overloaded to do stuff before a step
            self.pre_computation_hook()

            self.set(cdt.START_ITERATION_A2C, True)
            self.logger.debug("Sent 'StartIteration' True.")

        # value is False on convergence/iteration limit
        else:
            self._finish_optimization()

    @abc.abstractmethod
    def get_new_measurement(self):
        """
        Retrieve new measurement from relevant sensors
        """
        pass

    @abc.abstractmethod
    def _finish_optimization(self):
        """
        Finalize an iteration. Usually, this includes setting the actuation.
        """
        pass

    @abc.abstractmethod
    def optimize(self, variable: AgentVariable):
        """
        Performs the optimization given the information from the coordinator.
        Replies with local information.
        """
        pass

    @abc.abstractmethod
    def shift_trajectories(self):
        """
        Shifts algorithm specific trajectories.
        """
        pass

    @abc.abstractmethod
    def registration_callback(self, variable: AgentVariable):
        """
        Handles registration with the coordinator.
        """
        pass
