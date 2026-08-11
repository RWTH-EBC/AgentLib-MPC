"""
This is the predictor module. Its purpose is to provide random variables to the MPC agent. Its a manually written module and 
is referenced in the according config.json file (here: examples/getting-started/predictor/config.json). There are three parameters
that influence the prediction: time_step, prediction_horizon and sampling_time. time_step determines the time interval between predictions,
prediction_horizon specifies the number of future time steps to predict, and sampling_time controls how often a new prediction is generated.
All three parameters therefore also influence the prediction that will be generated in the MPC module.

Further random variables can be added by defining them in the outputs field and referencing them in the config.json
which uses this module. Optionally, the variable can be changed dynamically by changing the value in the process function.

The solar radiation is already implemented as such a variable. It is declared as an output (Q_sol) and set dynamically
in the process function. The solar radiation in the fmu is given by a function that is equivalent to:

q_sol = 5 + 10 * np.sin(2 * np.pi * grid / 86400 - np.pi / 2)

"""


import agentlib as al
import numpy as np
import pandas as pd
from typing import List


class PredictorModuleConfig(al.BaseModuleConfig):
    """Module that outputs a prediction of the heat load at a specified interval."""

    outputs: al.AgentVariables = [
        al.AgentVariable(
            name="T_amb", description="Ambient temperature"
        ),
        al.AgentVariable(
            name="T_upper", description="Upper temperature limit"
        ),
        al.AgentVariable(
            name="T_lower", description="Lower temperature limit"
        ),
        al.AgentVariable(
            name="Q_sol", description="Solar heat gain"
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
    ]

    shared_variable_fields: List[str] = ["outputs"]

class PredictorModule(al.BaseModule):
    """Module that outputs a prediction of the heat load at a specified interval."""

    config: PredictorModuleConfig

    def register_callbacks(self):
        pass

    def process(self):
        """Sets a new prediction at each time step."""
        while True:
            ts = self.get("time_step").value
            n = self.get("prediction_horizon").value
            now = self.env.now
            sample_time = self.get("sampling_time").value

            # Temperature prediction
            grid = np.arange(now, now + n * ts, ts)

            amb = 278.15 + 5 * np.sin(2 * np.pi * grid / 86400)
            self.set("T_amb", pd.Series(amb, index=list(grid)))

            # Solar radiation prediction
            q_sol = 5 + 10 * np.sin(2 * np.pi * grid / 86400 - np.pi / 2)
            self.set("Q_sol", pd.Series(q_sol, index=list(grid)))

            yield self.env.timeout(sample_time)
