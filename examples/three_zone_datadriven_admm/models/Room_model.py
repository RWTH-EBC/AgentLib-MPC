from typing import List

from agentlib_mpc.models.casadi_model import (
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
)

from agentlib_mpc.models.casadi_ml_model import CasadiMLModel, CasadiMLModelConfig


class RoomCCAConfig(CasadiMLModelConfig):
    inputs: List[CasadiInput] = [
        CasadiInput(
            name="T_v", value=293.15, unit="K", description="Vorlauf temperatur"
        ),
        CasadiInput(name="T_ahu", value=293.15, unit="K", description="Temperatur AHU"),
        CasadiInput(
            name="mDot", value=0.1, unit="K", description="water mass flow into BKA"
        ),
        CasadiInput(
            name="mDot_ahu",
            value=0.025,
            unit="K",
            description="air water mass flow into AHU",
        ),
        # disturbances
        CasadiInput(name="d", value=0, unit="W", description="Heat load into zone"),
        CasadiInput(
            name="T_amb", value=299, unit="K", description="ambient temperature"
        ),
        CasadiInput(name="Q_rad", value=0, unit="W/m²", description="Radiation"),
        # settings
        CasadiInput(
            name="T_set",
            value=298.55,
            unit="K",
            description="Set point for T in objective function",
        ),
        CasadiInput(
            name="T_upper", value=301.15, unit="K", description="Upper boundary for T."
        ),
        CasadiInput(
            name="T_lower", value=288.15, unit="K", description="Upper boundary for T."
        ),
    ]
    states: List[CasadiState] = [
        # differential
        CasadiState(
            name=f"T_air", value=290.15, unit="K", description="Temperature of zone"
        ),
        CasadiState(
            name="T_CCA_0", value=293.15, unit="K", description="Temperatur der BKA"
        ),
        # slacks
        CasadiState(
            name="T_slack", value=0, unit="K", description="Temperatur der BKA"
        ),
    ]
    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="q_T",
            value=1,
            unit="-",
            description="Weight for T in objective function",
        ),
        CasadiParameter(
            name="s_T",
            value=1,
            unit="-",
            description="Weight for T in objective function",
        ),
    ]
    outputs: List[CasadiOutput] = [
        CasadiOutput(
            name="T_CCA_out", value=293.15, unit="K", description="Temperatur der BKA"
        ),
        CasadiOutput(
            name="T_air_out", value=293.15, unit="K", description="Temperatur der Luft"
        ),
    ]


class RoomCCA(CasadiMLModel):
    config: RoomCCAConfig

    def setup_system(self):
        # Define ae
        self.T_CCA_out.alg = self.T_CCA_0
        self.T_air_out.alg = self.T_air

        self.constraints = [
            (self.T_lower, self.T_air + self.T_slack, self.T_upper),
        ]

        # Objective function
        objective = sum(
            [
                1 * 10 * self.q_T * (self.T_air - self.T_set) ** 2,
                1 * 10 * self.s_T * self.T_slack**2,
            ]
        )

        return objective
