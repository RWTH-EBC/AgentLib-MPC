"""Model of a room, whose temperature is predicted by a recurrent ML-model."""

from typing import List

from agentlib_mpc.models.casadi_model import (
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
)
from agentlib_mpc.models.casadi_ml_model import CasadiMLModel, CasadiMLModelConfig


class RecurrentRoomModelConfig(CasadiMLModelConfig):
    inputs: List[CasadiInput] = [
        # control
        CasadiInput(name="mDot", value=0.02, unit="kg/s", description="Air mass flow"),
        # disturbances
        CasadiInput(name="load", value=150, unit="W", description="Heat load"),
        CasadiInput(
            name="T_in", value=290.15, unit="K", description="Inflow air temperature"
        ),
        # settings
        CasadiInput(
            name="T_upper", value=400, unit="K", description="Upper boundary for T"
        ),
    ]
    states: List[CasadiState] = [
        CasadiState(name="T", value=293.15, unit="K", description="Room temperature"),
        CasadiState(name="T_slack", value=0, unit="K", description="Slack for T"),
    ]
    parameters: List[CasadiParameter] = [
        CasadiParameter(name="s_T", value=1, unit="-", description="Weight for slack"),
        CasadiParameter(name="r_mDot", value=1, unit="-", description="Weight for mDot"),
    ]
    outputs: List[CasadiOutput] = [
        CasadiOutput(name="T_out", unit="K", description="Room temperature")
    ]


class RecurrentRoomModel(CasadiMLModel):
    config: RecurrentRoomModelConfig

    def setup_system(self):
        self.T_out.alg = self.T

        self.constraints = [
            (0, self.T + self.T_slack, self.T_upper),
        ]

        return sum([self.r_mDot * self.mDot, self.s_T * self.T_slack**2])
