import logging
from dataclasses import asdict
import abc

from pydantic import Field

from agentlib.core import (
    AgentVariable,
    Agent,
    AgentVariables,
)
from agentlib.core.datamodels import Source
from agentlib_mpc.data_structures.coordinator_datatypes import RegistrationMessage
import agentlib_mpc.data_structures.coordinator_datatypes as cdt
from agentlib_mpc.modules.mpc.skippable_mixin import (
    SkippableMixinConfig,
    SkippableMixin,
)

logger = logging.getLogger(__name__)


class MiniEmployeeConfig(SkippableMixinConfig):
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


class MiniEmployee(SkippableMixin):
    config: MiniEmployeeConfig

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        self._registered_coordinator: Source = None
        self._start_optimization_at: float = 0
        self._result_obtained = False

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
        should_step_be_skipped = self.check_if_should_be_skipped()

        if should_step_be_skipped:
            self.logger.debug("DMPC participant was externally told to to skip.")
            self.set(cdt.START_ITERATION_A2C, False)
            return

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
            if self._result_obtained:
                self._finish_optimization()
                self._result_obtained = False

    def get_new_measurement(self):
        """
        Retrieve new measurement from relevant sensors
        Returns:

        """
        ...
        # raise NotImplementedError

        # return self.collect_variables_for_optimization()

    @abc.abstractmethod
    def _finish_optimization(self):
        """
        Finalize an iteration. Usually, this includes setting the actuation.
        Returns:

        """

    @abc.abstractmethod
    def optimize(self, variable: AgentVariable):
        """
        Performs the optimization given the information from the coordinator.
        Replies with local information.
        Returns:

        """
        variables = cdt.OptimizationData.from_dict(variable.value)

        # perform optimization
        # send optimizationData back to coordinator to signal finished
        # optimization

        value = variables.to_dict()
        self.logger.debug("Sent optimal solution.")
        self._result_obtained = True
        self.set(name=cdt.OPTIMIZATION_A2C, value=value)

    def shift_trajectories(self):
        """
        Shifts algorithm specific trajectories.
        Returns:

        """
        ...
        # raise NotImplementedError

    @abc.abstractmethod
    def registration_callback(self, variable: AgentVariable):
        """callback for registration"""
        self.logger.info(
            f"receiving {variable.name}={variable.value} from {variable.source}"
        )
        # global parameters to define optimisation problem
        value = RegistrationMessage(**variable.value)

        # Decide if message from coordinator is for this agent
        if not (value.agent_id == self.source.agent_id):
            return

        self.OptimOpts = value.opts
        status = True
        answer = RegistrationMessage(status=cdt.AgentStatus.standby)
        self._registered_coordinator = variable.source
        if status:
            self.set("registrationOut", asdict(answer))
