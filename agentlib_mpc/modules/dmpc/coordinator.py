import logging
import time
from dataclasses import asdict
from typing import Dict
import threading

from pydantic import Field

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

    maxIter: int = Field(default=10, description="Maximum number of iterations")
    time_out: float = Field(
        default=1, description="Maximum wait time for subsystems in seconds"
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


class Coordinator(BaseModule):
    """Class implementing the base coordination for distributed MPC"""

    config: CoordinatorConfig

    def __init__(self, *, config: dict, agent: Agent):
        super().__init__(config=config, agent=agent)
        self.AgentDict: Dict[Source, cdt.AgentDictEntry] = {}
        self.status: cdt.CoordinatorStatus = cdt.CoordinatorStatus.sleeping
        self.received_variable = threading.Event()

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
        for source, agent in self.AgentDict.items():
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

    def optim_results_callback(self, variable: AgentVariable):
        """
        Saves the results of a local optimization.
        Args:
            variable:

        Returns:

        """

        entry = self.AgentDict[variable.source]
        entry.optimization_data = cdt.OptimizationData.from_dict(variable.value)
        self.AgentDict[variable.source].status = cdt.AgentStatus.ready
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
            ag_dict_entry = self.AgentDict[variable.source]
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

    @property
    def all_finished(self):
        """

        Returns:
            True, if there are no busy agents, else False

        """
        for src, ag_entry in self.AgentDict.items():
            if ag_entry.status is cdt.AgentStatus.busy:
                return False
        return True

    def registration_callback(self, variable: AgentVariable):
        self.logger.info(
            f"receiving {variable.name}={variable.value} from {variable.source}"
        )
        # use information in message to set up coordinator

        if not (variable.source in self.AgentDict):  # add agent to dict
            entry = cdt.AgentDictEntry(
                name=variable.source,
                status=AgentStatus.pending,
            )
            self.AgentDict[variable.source] = entry
            OptimOpts = {"Nhor": 10, "dt": 60}
            message = RegistrationMessage(source=variable.source, opts=OptimOpts)
            self.set(cdt.REGISTRATION_C2A, asdict(message))  # {"source" :
            # variable.source, "status" : True, "opts" : OptimOpts}
            self.logger.info(
                f"Coordinator got request agent {variable.source} and set to "
                f"'pending'."
            )
        else:  # process ready-flag
            message = RegistrationMessage(**variable.value)
            if message.status == AgentStatus.standby:
                self.AgentDict[variable.source].status = AgentStatus.standby  #
                # change from
                # pending to ready
                self.logger.info(
                    f"Coordinator successfully registered agent {variable.source}."
                )
            else:
                self.AgentDict.pop(variable.source)  # delete agent from dict

    def _wait_for_ready(
        self,
    ):
        """Wait until all coupling variables arrive from the other systems."""

        self.received_variable.clear()
        self.logger.info("Start waiting for agents to finish computation.")
        while True:
            # check exit conditions
            if self.all_finished:
                count = 0
                for ag in self.AgentDict.values():
                    if ag.status == cdt.AgentStatus.ready:
                        count += 1
                self.logger.info("Got variables from all (%s) agents.", count)
                break

            # wait until a new item is put in the queue

            if self.received_variable.wait(timeout=self.config.time_out):
                self.received_variable.clear()
            else:
                self._deregister_slow_participants()
                break

    def _deregister_slow_participants(self):
        """Sets all agents that are still busy to standby, so they won't be
        waited on again."""
        for agent in self.AgentDict.values():
            if agent.status == cdt.AgentStatus.busy:
                agent.status = cdt.AgentStatus.standby


if __name__ == "__main__":
    pass
