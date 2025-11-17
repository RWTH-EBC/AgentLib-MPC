from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiInput,
    CasadiState,
    CasadiParameter,
    CasadiOutput,
    CasadiModelConfig,
)


class RLTConfig(CasadiModelConfig):
    inputs: list[CasadiInput] = [
        CasadiInput(
            name="mDot",
            value=0.0225,
            lb=0,
            ub=0.05,
            unit="kg/s",
            description="Air mass flow out of cooler.",
        )
    ]

    states: list[CasadiState] = []

    parameters: list[CasadiParameter] = [
        CasadiParameter(
            name="penalty",
            value=1,
            description="Penalty for air mass flow out of cooler.",
        )
    ]

    outputs: list[CasadiOutput] = [
        CasadiOutput(
            name="mDot_out",
            value=0.0225,
            lb=0,
            ub=0.05,
            unit="kg/s",
            description="Air mass flow out of cooler.",
        )
    ]


class RLT(CasadiModel):
    config: RLTConfig

    def setup_system(self):
        # Define ae

        self.mDot_out.alg = -self.mDot

        cost_function = self.penalty * self.mDot
        return cost_function
