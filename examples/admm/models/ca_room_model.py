from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)


class CaCooledRoomConfig(CasadiModelConfig):
    inputs: list[CasadiInput] = [
        # couplings
        CasadiInput(
            name="mDot_0",
            value=0.0225,
            unit="K",
            description="Air mass flow into zone 0",
        ),
        # disturbances
        CasadiInput(
            name="d_0", value=150, unit="W", description="Heat load into zone 0"
        ),
        CasadiInput(
            name="T_in", value=290.15, unit="K", description="Inflow air temperature"
        ),
        # settings
        CasadiInput(
            name="T_0_set",
            value=294.15,
            unit="K",
            description="Set point for T_0 in objective function",
        ),
        CasadiInput(
            name="T_0_upper",
            value=294.15,
            unit="K",
            description="Upper boundary (soft) for T_0.",
        ),
    ]

    states: list[CasadiState] = [
        # differential
        CasadiState(
            name=f"T_0", value=293.15, unit="K", description="Temperature of zone 0"
        ),
        # algebraic
        # slack variables
        CasadiState(
            name=f"T_0_slack",
            value=0,
            unit="K",
            description="Slack variable of temperature of zone 0",
        ),
    ]

    parameters: list[CasadiParameter] = [
        CasadiParameter(
            name="cp",
            value=1000,
            unit="J/kg*K",
            description="thermal capacity of the air",
        ),
        CasadiParameter(
            name="c_0",
            value=100000,
            unit="J/kg*K",
            description="thermal capacity of zone 0",
        ),
        CasadiParameter(
            name="q_T_0",
            value=1,
            unit="-",
            description="Weight for T_0 in objective function",
        ),
        CasadiParameter(
            name="s_T_0",
            value=1,
            unit="-",
            description="Weight for T_0 in constraint function",
        ),
    ]


class CaCooledRoom(CasadiModel):
    config: CaCooledRoomConfig

    def setup_system(self):
        # Define ode
        self.T_0.ode = (
            self.cp * self.mDot_0 / self.c_0 * (self.T_in - self.T_0)
            + self.d_0 / self.c_0
        )

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # soft constraints
            (0, self.T_0 + self.T_0_slack, self.T_0_upper),
            # outputs
        ]

        # Objective function
        objective = sum(
            [
                self.s_T_0 * self.T_0_slack**2,
                self.q_T_0 * (self.T_0 - self.T_0_set) ** 2,
            ]
        )

        return objective
