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
        # controls
        CasadiInput(
            name="Q_in",
            value=100,
            unit="W",
            description="Electrical power of heating rod",
        ),
        # disturbances
        CasadiInput(
            name="T_amb",
            value=290.15,
            unit="K",
            description="Ambient air temperature",
        ),
        # settings
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
        )
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
            name="Q_sol",
            value=0
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
        # Define ode
        self.T_zone.ode = (self.Q_in - self.U * (self.T_zone - self.T_amb) + self.Q_sol) / self.C

        # Define algebraic equation

        self.P_el.alg = ca.fabs(self.Q_in.sym)/self.COP  # casadi fabs = absolute value

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

