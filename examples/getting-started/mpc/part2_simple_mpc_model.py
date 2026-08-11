'''
In this example we will present different ways to define objective functions.
Check the setup_system function to see how objective functions are defined and returned.

The following objective types are demonstrated:
- CombinedObjective: sum of weighted SubObjectives
- ConditionalObjective: switch between objectives based on a condition (e.g. time)
- Nesting: combine objectives in arbitrary fashion
- ChangePenaltyObjective: penalize changes in a control variable

Each example is commented out; uncomment the one you want to use and return it.
'''


import logging
from typing import List
import matplotlib.pyplot as plt
from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)
from agentlib.utils.multi_agent_system import LocalMASAgency
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
        CasadiInput(
            name="Q_sol",
            value=0,
            unit="W",
            description="Solar radiation",
        ),
    ]

    states: List[CasadiState] = [
        CasadiState(
            name="T_zone",
            value=293.15,
            unit="K",
            description="Temperature of zone",
        ),
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
        ),
        CasadiParameter(
            name="switch",
            value=3600,
            unit="s",
            description="Time threshold for conditional objectives",
        )
    ]

    outputs: List[CasadiOutput] = [
        CasadiOutput(name="P_el", value=0)
    ]


class SimpleRoom(CasadiModel):

    config: SimpleRoomModelConfig

    def setup_system(self):
        self.T_zone.ode = (
            self.Q_in - self.U * (self.T_zone - self.T_amb) + self.Q_sol
        ) / self.C

        self.P_el.alg = ca.fabs(self.Q_in.sym)/self.COP

        self.constraints = [

            (-inf, self.T_zone - self.T_slack, self.T_upper),
            (self.T_lower, self.T_zone + self.T_slack, inf),
            (0, self.T_slack, inf),

            (-100, self.Q_in, 200),
            (0, self.P_el, inf)
        ]



        # CombinedObjective:
        #
        # In part 1 of the tutorial, the objective was defined as
        # 
        #   objective = sum(
        #       [
        #           self.T_slack ** 2 * self.s_T,
        #           self.P_el * self.r_pel,
        #       ]
        #   )
        #
        # This notation however is deprecated (but still supported).
        # The objective function should rather be defined as a CombinedObjective for example. The same objective as above can be
        # defined as a CombinedObjective, which simply sums up different terms with their respective weights. 
        # For that, these terms get created as SubObjective objects and then combined via a CombinedObjective:

        obj_slack_basic = self.create_sub_objective(
            expressions=self.T_slack**2,
            weight=self.s_T,
            name="temp_slack_basic",
        )
        obj_power_basic = self.create_sub_objective(
            expressions=self.P_el,
            weight=self.r_pel,
            name="power_basic",
        )
        combined_objective_basic = self.create_combined_objective(
            obj_slack_basic,
            obj_power_basic,
            normalization=1,
        )
        # return combined_objective_basic





        # ConditionalObjective:
        #
        # A ConditionalObjective can switch between different combined objectives, according to a condition.
        # Here, we penalize slack more heavily after 12 hours (43200 seconds).
        # For that, we define a second combined objective with higher weight on the slack and then switch between the two objectives based on the time.
        obj_power = self.create_sub_objective(
            expressions=self.P_el,
            weight=self.r_pel,
            name="power",
        )
        obj_slack = self.create_sub_objective(
            expressions=self.T_slack**2,
            weight=self.s_T,
            name="temp_slack",
        )
        obj_slack_high = self.create_sub_objective(
            expressions=self.T_slack**2,
            weight=self.s_T,
            name="temp_slack_high",
        )
        obj_slack_high = obj_slack_high * 50

        # First Objective
        combined_objective = self.create_combined_objective(
            obj_slack,
            obj_power,
            normalization=1,
        )
        # Second objective
        combined_objective_high = self.create_combined_objective(
            obj_slack_high,
            obj_power,
            normalization=1,
        )
        # Conditional objective
        objective = self.create_conditional_objective(
            (self.time < 43200, combined_objective),
            default_objective=combined_objective_high,
        )
        # return objective





        # Nesting:
        #
        # In theory, objectives can be nested in arbitrary fashion. 
        # For example, we could define objectives like shown in the following.
        # Here, the combined objectives used for the final objective are trivial, but they could be more complex like in the examples above.
        obj2_power = self.create_sub_objective(
            expressions=self.P_el,
            weight=1,
            name="obj2_power",
        )
        obj2_slack = self.create_sub_objective(
            expressions=self.T_slack**2,
            weight=1,
            name="obj2_slack",
        )
        objective2_1 = self.create_combined_objective(obj2_power)
        objective2_2 = self.create_combined_objective(obj2_slack)
        objective2 = objective2_1 + objective2_2 * self.s_T

        condition = ca.if_else(
            self.time < self.switch.sym,
            ca.if_else(
                self.time > 100,
                1,
                ca.if_else(self.time == 0, 0, 1),
            ),
            ca.if_else(self.time < 300, 0, 0),
        ) > 0

        nested_objective_example = self.create_conditional_objective(
            (condition, combined_objective_basic),
            default_objective=objective2,
        )
        # return nested_objective_example





        # ChangePenaltyObjective:
        #
        # Besides these classic objectives, you can define a ChangePenaltyObjective to penalize changes in a control variable.
        # This creates a specific subtype of SubObjective and therefore can be used in a CombinedObjective or ConditionalObjective just like the other SubObjectives.
        # Here, we penalize changes in the control input Q_in to avoid aggressive changes in the control trajectory.
        delta_qin = self.create_change_penalty(
            expressions=self.Q_in,
            weight=1,
            name="delta_qin",
        )
        # A ChangePenaltyObjective is a SubObjective, so it must be wrapped in a
        # CombinedObjective before being returned (like any other SubObjective).
        combined_with_delta = self.create_combined_objective(
            delta_qin,
            normalization=1,
        )
        # return combined_with_delta

        return objective