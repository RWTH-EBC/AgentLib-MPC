"""Module which generates random set points within a comfort zone. Code heavily stolen
from Max Berktold"""

import datetime
import random

from agentlib.core import BaseModuleConfig, BaseModule, Agent, AgentVariable


class SetPointGeneratorConfig(BaseModuleConfig):
    """
    Pydantic data model for ANNTrainer configuration parser
    """

    target_variable: AgentVariable = AgentVariable(name="target")
    day_start: int = 8
    day_end: int = 16
    day_lb: float = 273.15 + 19
    night_lb: float = 273.15 + 16
    day_ub: float = 273.15 + 21
    night_ub: float = 273.15 + 24
    interval: int = 60 * 60 * 4
    shared_variable_fields: list[str] = ["target_variable"]


class SetPointGenerator(BaseModule):
    """
    Module that generates and sends random set points based on daytime and values.
    """

    config: SetPointGeneratorConfig

    def __init__(self, config: dict, agent: Agent):
        """
        Constructor for model predictive controller (MPC).
        """
        super().__init__(config=config, agent=agent)
        self.last_randomization: float = self.env.time
        lb, ub = self._bounds()
        self.current_target = random.uniform(lb, ub)

    def register_callbacks(self): ...

    def process(self):
        while True:
            self.update_target()
            self.set(self.config.target_variable.name, self.current_target)
            yield self.env.timeout(self.config.interval)

    def update_target(self):
        """Updates the control target for a given time"""

        time = self.env.time
        lb, ub = self._bounds()

        # update target, if enough time has passed or the target violates boundaries
        if (
            time - self.last_randomization >= self.config.interval
            or self.current_target < lb
            or ub < self.current_target
        ):
            self.current_target = random.uniform(lb, ub)
            self.last_randomization = time
            self.logger.debug(
                f"Set target {self.config.target_variable.name} to "
                f"{self.current_target:.2f} {self.config.target_variable.unit}"
            )

    def _bounds(self) -> tuple[float, float]:
        """Returns the lower and upper bound for a given time"""

        if self._is_weekend():
            return self.config.night_lb, self.config.night_ub

        if self._is_daytime():
            return self.config.day_lb, self.config.day_ub

        return self.config.night_lb, self.config.night_ub

    def _is_daytime(self) -> bool:
        """Returns True if the given time is during day"""

        time = datetime.datetime.fromtimestamp(self.env.time)

        return self.config.day_start <= time.hour <= self.config.day_end

    def _is_weekend(self) -> bool:
        """returns True if the given time is during weekend"""

        time = datetime.datetime.fromtimestamp(self.env.time)

        return 5 <= time.weekday()
