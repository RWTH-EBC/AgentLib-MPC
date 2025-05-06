from agentlib import AgentVariables
from agentlib.core import BaseModule, BaseModuleConfig, AgentVariable
from pydantic import Field
from typing import Optional

from agentlib_mpc import utils
from agentlib_mpc.data_structures import mpc_datamodels


class MPCOnOffConfig(BaseModuleConfig):
    active: AgentVariable = Field(
        default=AgentVariable(
            name=mpc_datamodels.MPC_FLAG_ACTIVE,
            description="MPC is active",
            type="bool",
            value=True,
            shared=False,
        ),
        description="Variable used to activate or deactivate the MPC operation",
    )
    inputs: AgentVariables = Field(
        default=[], description="Inputs based on which switch decisions can be made."
    )
    t_sample: float = Field(
        default=60, description="Sends the active variable every other t_sample"
    )
    public_active_message: Optional[AgentVariable] = Field(
        default=None,
        description="If needed, specify an AgentVariable that is sent when the MPC is active, for example to suppress a local controller.",
    )
    public_inactive_message: Optional[AgentVariable] = Field(
        default=None,
        description="If needed, specify an AgentVariable that is sent when the MPC is inactive, for example to awaken a local controller.",
    )
    controls_when_deactivated: AgentVariables = Field(
        default=[], description="List of AgentVariables to send as Fallback Controls."
    )

    shared_variable_fields: list[str] = [
        "public_active_message",
        "controls_when_deactivated",
    ]


class MPCOnOff(BaseModule):
    config: MPCOnOffConfig

    def process(self):
        while True:
            deactivate_mpc = self.check_mpc_deactivation()
            if deactivate_mpc:
                self.deactivate_mpc()
            else:
                self.activate_mpc()
            yield self.env.timeout(self.config.t_sample)

    def deactivate_mpc(self):
        """Performs mpc deactivation. Sends the deactivation signal, as well as
        default control signals."""
        self.set(self.config.active.name, False)
        for agent_variable in self.config.controls_when_deactivated:
            # actively resend this variable
            self.set(agent_variable.name, agent_variable.value)
        if self.config.public_inactive_message is not None:
            self.set(
                self.config.public_inactive_message.name,
                self.config.public_inactive_message.value,
            )

    def activate_mpc(self):
        """Performs mpc activation. Sends activation signal, as well as the public
        active message."""
        self.set(self.config.active.name, True)
        if self.config.public_active_message is not None:
            self.set(
                self.config.public_active_message.name,
                self.config.public_active_message.value,
            )

    def check_mpc_deactivation(self) -> bool:
        """This function can be overridden, to define conditions based on which an
        MPC module within this agent should be deactivated. Returns True if MPC
        should be deactivated, and False if it should be active."""

    def register_callbacks(self):
        """This function can be overridden to check the deactivation in an
        event-based manner."""


class SkipMPCInIntervalsConfig(MPCOnOffConfig):
    """
    Config for a module which deactivates any MPC by sending the variable
    `active` in the specified intervals.
    """

    intervals: list[tuple[float, float]] = Field(
        default=[], description="If environment time is within these intervals"
    )
    time_unit: utils.TimeConversionTypes = Field(
        default="seconds",
        description="Specifies the unit of the given "
        "`skip_mpc_in_intervals`, e.g. seconds or days.",
    )


class SkipMPCInIntervals(MPCOnOff):
    """
    Module which deactivates any MPC by sending the variable
    `active` in the specified intervals.
    """

    config: SkipMPCInIntervalsConfig

    def check_mpc_deactivation(self) -> bool:
        if utils.is_time_in_intervals(
            time=self.env.time / utils.TIME_CONVERSION[self.config.time_unit],
            intervals=self.config.intervals,
        ):
            self.logger.debug(
                "Current time is in skip_mpc_in_intervals, sending active=False to MPC"
            )
            return True
        return False
