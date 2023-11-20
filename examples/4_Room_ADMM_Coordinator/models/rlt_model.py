from agentlib_mpc.models.casadi_model import *

ROOM_NUMBER = 4

mass_flow_inputs = [
    CasadiInput(
        name=f"mDot_{i+1}",
        value=0.0225,
        unit="kg/s",
        description=f"Air mass flow out of cooler {i}.",
    )
    for i in range(ROOM_NUMBER)
]

mass_flow_outputs = [
    CasadiOutput(
        name=f"mDot_out_{i+1}",
        value=0.0225,
        unit="kg/s",
        description=f"Air mass flow out of cooler {i}.",
    )
    for i in range(ROOM_NUMBER)
]


class RLTConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = mass_flow_inputs

    states: List[CasadiState] = []

    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="mDot_max",
            value=0.075,
            unit="-",
            description="Maximum air mass flow possible.",
        )
    ]

    outputs: List[CasadiOutput] = mass_flow_outputs


class RLT(CasadiModel):
    config: RLTConfig

    def setup_system(self):
        # Define ae
        self.mDot_out_1.alg = 1 * self.mDot_1
        self.mDot_out_2.alg = 1 * self.mDot_2
        self.mDot_out_3.alg = 1 * self.mDot_3
        self.mDot_out_4.alg = 1 * self.mDot_4

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            (0, self.mDot_1 + self.mDot_2 + self.mDot_3 + self.mDot_4, self.mDot_max)
            # soft constraints
            # outputs
        ]

        return 0
