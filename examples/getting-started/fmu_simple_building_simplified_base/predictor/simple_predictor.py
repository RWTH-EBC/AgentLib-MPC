import agentlib as al
import numpy as np
import pandas as pd
from typing import List


class PredictorModuleConfig(al.BaseModuleConfig):
    """Module that outputs a prediction of the heat load at a specified
    interval."""

    outputs: al.AgentVariables = [
        al.AgentVariable(
            name="T_amb", description="test_description"
        ),
        al.AgentVariable(
            name="T_upper", description="test_description"
        ),
        al.AgentVariable(
            name="T_lower", description="test_description"
        ),
    ]

    parameters: al.AgentVariables = [
        al.AgentVariable(
            name="time_step", value=900, description="Sampling time for prediction."
        ),
        al.AgentVariable(
            name="prediction_horizon",
            value=10,
            description="Number of sampling points for prediction.",
        ),
        al.AgentVariable(
            name="sampling_time",
            value=10,
            description="Time between prediction updates",
        ),
        al.AgentVariable(
            name="comfort_interval",
            value=86400,
            description="Time between comfort updates.",
        ),
        al.AgentVariable(
            name="upper_comfort_high",
            value=299,
            description="High value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="upper_comfort_low",
            value=297,
            description="Low value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="lower_comfort_high",
            value=292,
            description="High value in the comfort set point trajectory.",
        ),
        al.AgentVariable(
            name="lower_comfort_low",
            value=290,
            description="Low value in the comfort set point trajectory.",
        ),
    ]

    shared_variable_fields: List[str] = ["outputs"]
    #shared_variable_fields = ["outputs"]


class PredictorModule(al.BaseModule):
    """Module that outputs a prediction of the heat load at a specified
    interval."""

    config: PredictorModuleConfig

    def register_callbacks(self):
        pass

    def process(self):
        """Sets a new prediction at each time step."""
        self.env.process(self.send_upper_comfort_trajectory())
        self.env.process(self.send_lower_comfort_trajectory())

        while True:
            ts = self.get("time_step").value
            n = self.get("prediction_horizon").value
            now = self.env.now
            sample_time = self.get("sampling_time").value

            # temperature prediction
            grid = np.arange(now, now + n * ts, ts)
            values = amb_temp_func(grid, uncertainty=0)
            traj = pd.Series(values, index=list(grid))
            self.set("T_amb", traj)
            yield self.env.timeout(sample_time)

    def send_upper_comfort_trajectory(self):
        """Sends the series for the comfort condition."""
        while True:
            now = self.env.now
            comfort_interval = self.get("comfort_interval").value

            # temperature prediction
            grid = np.arange(now, now + 2 * comfort_interval, 0.5 * comfort_interval)
            values = [self.get("upper_comfort_high").value, self.get("upper_comfort_low").value] * 2
            traj = pd.Series(values, index=list(grid))
            self.set("T_upper", traj)
            yield self.env.timeout(comfort_interval)

    def send_lower_comfort_trajectory(self):
        """Sends the series for the comfort condition."""
        while True:
            now = self.env.now
            comfort_interval = self.get("comfort_interval").value

            # temperature prediction
            grid = np.arange(now, now + 2 * comfort_interval, 0.5 * comfort_interval)
            values = [self.get("lower_comfort_low").value, self.get("lower_comfort_high").value] * 2
            traj = pd.Series(values, index=list(grid))
            self.set("T_lower", traj)
            yield self.env.timeout(comfort_interval)


def amb_temp_func(current, uncertainty):
    """Returns the ambient temperature in K, given a time in seconds."""
    value = np.zeros(shape=current.shape)
    for i in range(current.size):
        random_factor = 1 + uncertainty * (np.random.random() - 0.5)
        value[i] = random_factor * (278.15 + 5 * np.sin(2*np.pi * current[i] / 86400))
    return value
