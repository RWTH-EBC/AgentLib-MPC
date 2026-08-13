'''
This is the MPC model module for part three of the getting-started.
The model itself is identical to part 2 (see mpc/part2_simple_mpc_model.py).
The focus of this part is on the advanced solver and discretization settings,
which are configured in mpc/part3_config.py rather than in this model file.
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

        return combined_objective_basic