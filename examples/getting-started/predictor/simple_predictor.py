"""
This is the predictor module. Its purpose is to provide random variables to the MPC agent. 
Further random variables could be added by defining them in the outputs field. 
You also need to reference them in the config.json which uses this module.
Optionally, the variable can be changed dynamically by changing the value in the process function.

For example, the solar energy could be added as an additional variable:

    al.AgentVariable(
        name="Q_sol", description="Solar radiation in W"
    ),

then you need to reference it in the config that will use this module (here: examples/getting-started/predictor/config.json).
There you could, for example, add this:

    "outputs": [
        "T_amb",
        "T_upper",
        "T_lower",
        "Q_sol"
    ],

If you want q_sol to be dynamic, you have to set it in the process function, just like the other variables.
The solar radiation in the fmu is given by a function that is equivalent to:

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

            yield self.env.timeout(sample_time)
