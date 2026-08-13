'''
This is the MPC model module. Its purpose is to define the system dynamics, constraints, 
and objective that the MPC agent optimizes. A model prective control setup consists of a process model and an optimal control problem. 
The process model is the internal model that is used to predict the future system behavior, while the optimal control problem defines the constraints
and objective function that the MPC agent optimizes.

Constraints can be hard or soft.
Hard constraints: represented directly as bounded tuples `(lower, expression, upper)` without any slack (for
example `(-100, self.Q_in, 200)`). The solver must satisfy these exactly. 
Soft constraints: implemented by introducing a slack variable (an algebraic `CasadiState`)
so the inequality can be relaxed, e.g. `self.T_zone - self.T_slack <= self.T_upper`. Constrain the slack (e.g. `0 <= self.T_slack`) and
add a penalty term to the objective function. Violations are allowed but penalized.

The objective function is a weighted sum of the slack variables and other terms. It can be defined in different ways.
In a following part of this tutorial, they will be demonstrated in more detail.
All of the above characteristics are defined in the `setup_system` method of the model class (in this file).

Further variables can be added to the model by defining them in the `inputs`, `states`, `parameters`, or `outputs` fields 
of the model config class. You also need to reference them in the config.json file that uses this module
(here: examples/getting-started/mpc/part1_config.json).

As you can see in the fmu, the model features a solar radiation input. The internal process model of the mpc
does not yet account for that. Therefore, the solar energy should be added as an additional input. For example, like:

    CasadiInput(
        name="Q_sol", 
        value=0, 
        unit="W", 
        description="Solar radiation"
    ),

Then you need to reference it in the config.json that uses this module (here: examples/getting-started/mpc/part1_config.json). 
There you could, for example, add it to the inputs list like this:

    "inputs": [
        {"name": "T_upper", "value": 298, "interpolation_method": "previous"},
        {"name": "T_lower", "value": 296, "interpolation_method": "previous"},
        {"name": "T_amb", "value": 278.15},
        {"name": "Q_sol", "value": 0, "interpolation_method": "previous"}
    ],

To add q_sol to the internal model you would have to modify the according ode that is specified in the `setup_system` method
in the model class. For example you could replace it by:

    self.T_zone.ode = (self.Q_in - self.U * (self.T_zone - self.T_amb) + self.Q_sol) / self.C

which will mirror the way the solar radiation is accounted for in the fmu. Run the simulation again and see how
the MPC agent now accounts for the solar radiation in its predictions and control actions.

'''




import logging
from typing import List
from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from math import inf
import casadi as ca

logger = logging.getLogger(__name__)


class SimpleRoomModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        CasadiInput(
            name="Q_in",
            value=100,
            unit="W",
            description="Electrical power of heating rod",
        ),
        CasadiInput(
            name="T_amb",
            value=290.15,
            unit="K",
            description="Ambient air temperature",
        ),
        CasadiInput(
            name="T_upper",
            value=294.15,
            unit="K",
            description="Upper boundary (soft) for T.",
        ),
        CasadiInput(
            name="T_lower",
            value=290.15,
            unit="K",
            description="Lower boundary (soft) for T.",
        ),
    ]

    states: List[CasadiState] = [
        # differential
        CasadiState(
            name="T_zone",
            value=293.15,
            unit="K",
            description="Temperature of zone",
        ),
        # algebraic
        # slack variables
        CasadiState(
            name="T_slack",
            value=0,
            unit="K",
            description="Slack variable of upper temperature of zone",
        )
    ]

    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="C",
            value=10_000,
            unit="J/K",
            description="thermal capacity of zone",
        ),
        CasadiParameter(
            name="U",
            value=5,
            unit="W/K",
            description="thermal conductivity of zone",
        ),
        CasadiParameter(
            name="s_T",
            value=100,
            unit="-",
            description="Weight for T in upper constraint function",
        ),
        CasadiParameter(
            name="r_pel",
            value=1,
            unit="-",
            description="Weight for P_el in objective function",
        ),
        CasadiParameter(
            name="COP",
            value=3
        )
    ]

    outputs: List[CasadiOutput] = [
        CasadiOutput(name="P_el", value=0)
    ]


class SimpleRoom(CasadiModel):

    config: SimpleRoomModelConfig

    def setup_system(self):

        # -------------------
        # MPC Processmodel
        # -------------------

        # Define ode to represent the mpc's internal model of the system dynamics.
        self.T_zone.ode = (self.Q_in - self.U * (self.T_zone - self.T_amb)) / self.C

        # Define algebraic equation 
        self.P_el.alg = ca.fabs(self.Q_in.sym)/self.COP  # casadi fabs = absolute value

        # -------------------
        # Optimal Control Problem
        # -------------------

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [

            # soft constraints
            (-inf, self.T_zone - self.T_slack, self.T_upper),
            (self.T_lower, self.T_zone + self.T_slack, inf),
            (0, self.T_slack, inf),

            # hard constraints
            (-100, self.Q_in, 200),
            (0, self.P_el, inf)
        ]

        # Objective function
        objective = sum(
            [
                self.T_slack ** 2 * self.s_T,
                self.P_el * self.r_pel,
            ]
        )

        return objective

