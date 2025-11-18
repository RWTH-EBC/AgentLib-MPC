from agentlib_mpc.models.casadi_model import *


class CaCooledRoomConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        # couplings
        CasadiInput(
            name="mDot",
            value=0.0225,
            unit="kg/s",
            description="Air mass flow into zone",
        ),
        # disturbances
        CasadiInput(name="d", value=150, unit="W", description="Heat load into zone"),
        CasadiInput(
            name="T_in", value=290.15, unit="K", description="Inflow air temperature"
        ),
        # settings
        CasadiInput(
            name="T_set",
            value=294.15,
            unit="K",
            description="Set point for T in objective function",
        ),
        CasadiInput(
            name="T_upper", value=294.15, unit="K", description="Upper boundary for T."
        ),
    ]

    states: List[CasadiState] = [
        # differential
        CasadiState(
            name="T", value=293.15, unit="K", description="Temperature of zone"
        ),
        # algebraic
        # slack variables
    ]

    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="cp",
            value=1000,
            unit="J/kg*K",
            description="thermal capacity of the air",
        ),
        CasadiParameter(
            name="cZ",
            value=60000,
            unit="J/kg*K",
            description="thermal capacity of zone",
        ),
        CasadiParameter(
            name="q_T",
            value=1,
            unit="-",
            description="Weight for T in objective function",
        ),
        CasadiParameter(
            name="q_mDot",
            value=1,
            unit="-",
            description="Weight for mDot in objective function",
        ),
    ]


class CaCooledRoom(CasadiModel):
    config: CaCooledRoomConfig

    def setup_system(self):
        # Define ode
        self.T.ode = (
            self.cp * self.mDot / self.cZ * (self.T_in - self.T) + self.d / self.cZ
        )

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # soft constraints
            (0, self.T, self.T_upper),
            # outputs
        ]

        # Objective function
        objective = sum(
            [
                0.0001 * self.q_T * (self.T - self.T_set) ** 2,
                0.0001 * self.q_mDot * (1 / 0.167) ** 2 * self.mDot**2,
            ]
        )
        return objective


class CaCooledRoomSimConfig(CaCooledRoomConfig):
    outputs: List[CasadiOutput] = [
        CasadiOutput(
            name="T_out", value=293.15, unit="K", description="Room temperature"
        ),
    ]


class CaCooledRoomSim(CaCooledRoom):
    config: CaCooledRoomSimConfig

    def setup_system(self):
        obj = super().setup_system()

        # Define dae
        self.T_out.alg = self.T

        return obj
