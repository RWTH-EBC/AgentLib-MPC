from agentlib_mpc.models.casadi_model import *


class AHUConfig(CasadiModelConfig):
    inputs: List[CasadiInput] = [
        CasadiInput(
            name="mDot_0",
            value=0.025,
            unit="K",
            description="Air mass flow into Lüftungsgerät",
        ),
        CasadiInput(
            name="T_amb", value=295, unit="K", description="ambient temperature"
        ),
        # T_ahu control variable
        CasadiInput(name="T_ahu1", value=294, unit="K", description="Temperatur AHU"),
        CasadiInput(name="T_ahu2", value=294, unit="K", description="Temperatur AHU"),
        CasadiInput(name="T_ahu3", value=294, unit="K", description="Temperatur AHU"),
        # T_raum
        CasadiInput(name="T_room1", value=299, unit="K", description="Raumtemperatur"),
        CasadiInput(name="T_room2", value=299, unit="K", description="Raumtemperatur"),
        CasadiInput(name="T_room3", value=299, unit="K", description="Raumtemperatur"),
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
            name="cl", value=1000, unit="J/kg*K", description="Heat Capacity"
        ),
    ]

    outputs: List[CasadiOutput] = [
        CasadiOutput(
            name="T_ahu_out1",
            value=293,
            unit="K",
            description="Air mass flow out of cooler.",
        ),
        CasadiOutput(
            name="T_ahu_out2",
            value=293,
            unit="K",
            description="Air mass flow out of cooler.",
        ),
        CasadiOutput(
            name="T_ahu_out3",
            value=293,
            unit="K",
            description="Air mass flow out of cooler.",
        ),
        CasadiOutput(name="W1", value=200, unit="K", description="Leistung"),
        CasadiOutput(name="W2", value=200, unit="K", description="Leistung"),
        CasadiOutput(name="W3", value=200, unit="K", description="Leistung"),
    ]


class AHU(CasadiModel):
    config: AHUConfig

    def setup_system(self):
        self.T_ahu_out1.alg = 1 * self.T_ahu1
        self.T_ahu_out2.alg = 1 * self.T_ahu2
        self.T_ahu_out3.alg = 1 * self.T_ahu3
        self.W1.alg = (
            self.cl * self.mDot_0 * (self.T_ahu1 - (self.T_room1 + self.T_amb) / 2)
        )
        self.W2.alg = (
            self.cl * self.mDot_0 * (self.T_ahu2 - (self.T_room2 + self.T_amb) / 2)
        )
        self.W3.alg = (
            self.cl * self.mDot_0 * (self.T_ahu3 - (self.T_room3 + self.T_amb) / 2)
        )

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
                * (
                    (
                        self.cl
                        * self.mDot_0
                        * (self.T_ahu1 - (self.T_room1 + self.T_amb) / 2)
                    )
                    ** 2
                    + 0.02
                )
                ** 0.5,
                0.1
                * 0.001
                * self.r_T_v
                * (
                    (
                        self.cl
                        * self.mDot_0
                        * (self.T_ahu2 - (self.T_room2 + self.T_amb) / 2)
                    )
                    ** 2
                    + 0.02
                )
                ** 0.5,
                0.1
                * 0.001
                * self.r_T_v
                * (
                    (
                        self.cl
                        * self.mDot_0
                        * (self.T_ahu3 - (self.T_room3 + self.T_amb) / 2)
                    )
                    ** 2
                    + 0.02
                )
                ** 0.5,
            ]
        )

        return objective
