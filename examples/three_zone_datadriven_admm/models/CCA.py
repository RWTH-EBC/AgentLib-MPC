from agentlib_mpc.models.casadi_model import *


class TempControllerConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        CasadiInput(
            name="mDot_0", value=0.1, unit="K", description="mass flow into BKA"
        ),
        # controls
        CasadiInput(name="T_v", value=294, unit="K", description="Vorlauftemperatur"),
        CasadiInput(name="T_r1", value=296, unit="W", description="Rücklauf"),
        CasadiInput(name="T_r2", value=296, unit="W", description="Rücklauf"),
        CasadiInput(name="T_r3", value=296, unit="W", description="Rücklauf"),
    ]

    states: List[CasadiState] = []

    parameters: List[CasadiParameter] = [
        CasadiParameter(
            name="r_T_v",
            value=1,
            unit="-",
            description="Weight for mDot in objective function",
        ),
        CasadiParameter(
            name="cp", value=4200, unit="J/kg*K", description="Heat Capacity"
        ),
    ]

    outputs: List[CasadiOutput] = [
        CasadiOutput(
            name="T_v_out",
            value=293,
            unit="K",
            description="Air mass flow out of cooler.",
        ),
        CasadiOutput(
            name="T_v_out2",
            value=293,
            unit="K",
            description="Air mass flow out of cooler.",
        ),
        CasadiOutput(
            name="T_v_out3",
            value=293,
            unit="K",
            description="Air mass flow out of cooler.",
        ),
        CasadiOutput(name="W1", value=200, unit="K", description="Leistung"),
        CasadiOutput(name="W2", value=200, unit="K", description="Leistung"),
        CasadiOutput(name="W3", value=200, unit="K", description="Leistung"),
    ]


class TempController(CasadiModel):

    config: TempControllerConfig

    def setup_system(self):
        self.T_v_out.alg = 1 * self.T_v
        self.T_v_out2.alg = 1 * self.T_v
        self.T_v_out3.alg = 1 * self.T_v
        self.W1.alg = self.cp * self.mDot_0 * (self.T_v - self.T_r1)
        self.W2.alg = self.cp * self.mDot_0 * (self.T_v - self.T_r2)
        self.W3.alg = self.cp * self.mDot_0 * (self.T_v - self.T_r3)

        # Constraints: List[(lower bound, function, upper bound)]
        self.constraints = [
            # soft constraints
        ]
        # Objective function
        objective = sum(
            [
                0.1
                * 0.001
                * self.r_T_v
                * ((self.cp * self.mDot_0 * (self.T_v - self.T_r1)) ** 2 + 0.02) ** 0.5,
                0.1
                * 0.001
                * self.r_T_v
                * ((self.cp * self.mDot_0 * (self.T_v - self.T_r2)) ** 2 + 0.02) ** 0.5,
                0.1
                * 0.001
                * self.r_T_v
                * ((self.cp * self.mDot_0 * (self.T_v - self.T_r3)) ** 2 + 0.02) ** 0.5,
            ]
        )

        return objective
