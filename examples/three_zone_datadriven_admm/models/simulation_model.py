"""
Currently simulating with white-box model;
For simulation with pure white-box model, activate all ODEs and change
CasadiMLModelConfig <-> CasadiModelConfig and
CasadiModelNetwork <-> CasadiModel
"""

from agentlib_mpc.models.casadi_model import (
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModel,
    CasadiModelConfig,
)
from typing import List


class FullModelConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        CasadiInput(name="T_v", value=294.15, unit="K", description="Boden Temperatur"),
        CasadiInput(name="T_ahu", value=295.15, unit="K", description="Temperatur AHU"),
        CasadiInput(name="mDot", value=0.1, unit="K", description="mass flow into BKA"),
        CasadiInput(
            name="mDot_ahu",
            value=0.025,
            unit="K",
            description="air water mass flow into AHU",
        ),
        # disturbances
        CasadiInput(name="d", value=400, unit="W", description="Heat load into zone"),
        CasadiInput(
            name="T_amb", value=283, unit="K", description="Aussenlufttemperatur"
        ),
        CasadiInput(name="Q_rad", value=300, unit="W/m²", description="Radiation"),
        # settings
        CasadiInput(
            name="T_set",
            value=294.55,
            unit="K",
            description="Set point for T in objective function",
        ),
        CasadiInput(
            name="T_upper", value=301.15, unit="K", description="Upper boundary for T."
        ),
        CasadiInput(
            name="T_lower", value=290.15, unit="K", description="Upper boundary for T."
        ),
    ]
    states: List[CasadiState] = [
        # differential
        CasadiState(
            name="T_CCA_0", value=295.15, unit="K", description="Temperatur der BKA"
        ),
        CasadiState(
            name=f"T_air", value=296, unit="K", description="Temperature of zone"
        ),
        CasadiState(
            name=f"T_wall",
            value=294.15,
            unit="K",
            description="Temperature of the wall",
        ),
    ]
    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="cp",
            value=4200,
            unit="J/kg*K",
            description="thermal capacity of medium",
        ),
        CasadiParameter(
            name="c_BKA",
            value=500000,
            unit="J/kg*K",
            description="thermal capacity of zone 0",
        ),
        CasadiParameter(
            name="cw", value=518000, unit="J/kg*K", description="Wärmekapazität Wand"
        ),
        CasadiParameter(
            name="cl", value=1000, unit="J/kg*K", description="Wärmekapazität Luft"
        ),
        CasadiParameter(
            name="hw", value=0.17, unit="J/kg*K", description="Leitfähigkeit Wand"
        ),
        CasadiParameter(
            name="hBKA", value=2, unit="J/kg*K", description="Leitfähigkeit BKA"
        ),
        CasadiParameter(
            name="hFenster",
            value=1.23,
            unit="J/kg*K",
            description="Leitfähigkeit Fenster",
        ),
        CasadiParameter(name="Aw", value=13.85, unit="m2", description="Fläche Wand"),
        CasadiParameter(name="ABKA", value=39.5, unit="m2", description="Fläche BKA"),
        CasadiParameter(
            name="AFenster", value=6.6, unit="m2", description="Fläche Fenster"
        ),
        CasadiParameter(
            name="mRoom", value=60, unit="kg", description="Luftmasse Raum"
        ),
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
            name="T_wall_out",
            value=290.15,
            unit="kg/s",
            description="Temperatur der Raumluft",
        ),
        CasadiOutput(
            name="T_0_out",
            value=299.15,
            unit="kg/s",
            description="Temperatur der Raumluft",
        ),
        CasadiOutput(
            name="T_CCA_out",
            value=299.15,
            unit="kg/s",
            description="Temperatur der Raumluft",
        ),
    ]


class FullModel(CasadiModel):

    config: FullModelConfig

    def setup_system(self):
        # Define ode
        self.T_CCA_0.ode = self.cp * self.mDot * (self.T_v - self.T_CCA_0) / (
            self.c_BKA * self.ABKA
        ) + self.hBKA / self.c_BKA * (self.T_air - self.T_CCA_0)
        self.T_wall.ode = (
            self.hw / self.cw * (self.T_air - self.T_wall)
            + self.hw / self.cw * (self.T_amb - self.T_wall)
            + self.Q_rad / self.cw
        )
        self.T_air.ode = (
            self.hw * self.Aw / (self.cl * self.mRoom) * (self.T_wall - self.T_air)
            + self.hBKA
            * self.ABKA
            / (self.cl * self.mRoom)
            * (self.T_CCA_0 - self.T_air)
            + self.d / (self.cl * self.mRoom)
            + (self.T_ahu - self.T_air) * self.mDot_ahu / self.mRoom
            + self.hFenster
            * self.AFenster
            / (self.cl * self.mRoom)
            * (self.T_amb - self.T_air)
            + self.Q_rad * self.AFenster / (self.cl * self.mRoom)
        )

        # Define ae
        self.T_0_out.alg = 1 * self.T_air
        self.T_wall_out.alg = 1 * self.T_wall
        self.T_CCA_out.alg = 1 * self.T_CCA_0

        objective = sum([])

        return objective
