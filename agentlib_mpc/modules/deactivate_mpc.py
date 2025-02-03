from agentlib.core import BaseModule, BaseModuleConfig, AgentVariable
from pydantic import Field

from agentlib_mpc import utils


class SkipMPCInIntervalsConfig(BaseModuleConfig):
    """
    Config for a module which deactivates any MPC by sending the variable
    `active` in the specified intervals.
    """
    intervals: list[tuple[float, float]] = Field(
        default=[],
        description="If environment time is within these intervals"
    )
    time_unit: utils.TimeConversionTypes = Field(
        default="seconds",
        description="Specifies the unit of the given "
                    "`skip_mpc_in_intervals`, e.g. seconds or days."
    )
    active: AgentVariable = Field(
        default=AgentVariable(name="active", description="MPC is active", type="bool", value=True, shared=False),
        description="Variable used to activate or deactivate the MPC operation"
    )
    t_sample: float = Field(
        default=60,
        description="Sends the active variable every other t_sample"
    )


class SkipMPCInIntervals(BaseModule):
    """
    Module which deactivates any MPC by sending the variable
    `active` in the specified intervals.
    """
    config: SkipMPCInIntervalsConfig

    def process(self):
        """Write the current data values into data_broker every t_sample"""
        while True:
            if utils.is_time_in_intervals(
                    time=self.env.now / utils.TIME_CONVERSION[self.config.time_unit],
                    intervals=self.config.intervals
            ):
                self.logger.debug("Current time is in skip_mpc_in_intervals, sending active=False to MPC")
                self.set("active", False)
            else:
                self.set("active", True)
            yield self.env.timeout(self.config.t_sample)

    def register_callbacks(self):
        """Don't do anything as this module is not event-triggered"""
