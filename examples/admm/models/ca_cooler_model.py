from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)


class CaCoolerConfig(CasadiModelConfig):
    inputs: list[CasadiInput] = [
        # controls
        CasadiInput(
            name="mDot",
            value=0.0225,
            unit="kg/s",
            description="Air " "mass flow out of cooler.",
        ),
        # disturbances
        # CasadiInput(name="T_oda", value=273.15 + 30, unit="K", description="Ambient air temperature"),
    ]

    states: list[CasadiState] = [
        # differential
        # algebraic
        # slack variables
    ]

    parameters: list[CasadiParameter] = [
        CasadiParameter(
            name="r_mDot",
            value=1,
            unit="-",
            description="Weight for mDot in objective function",
        )
    ]

    outputs: list[CasadiOutput] = [
        CasadiOutput(
            name="mDot_out",
            value=0.0225,
            unit="kg/s",
            description="Air mass flow out of cooler.",
        ),
    ]


class CaCooler(CasadiModel):
    config: CaCoolerConfig

    def setup_system(self):
        # Define ode

        # Define ae
        self.mDot_out.alg = 1 * self.mDot

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # soft constraints
            # outputs
        ]

        # Objective function
        objective = sum(
            [
                self.r_mDot * self.mDot,
            ]
        )

        return objective
