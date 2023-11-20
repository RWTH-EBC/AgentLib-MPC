from typing import List

from agentlib_mpc.models.casadi_model import (
    CasadiModel,
    CasadiModelConfig,
    CasadiInput,
    CasadiParameter,
    CasadiState,
    CasadiOutput,
)


class MyCasadiModelConfig(CasadiModelConfig):
    parameters: List[CasadiParameter] = [
        CasadiParameter(name="par", value=12, unit="kg", description="Test parameter"),
        CasadiParameter(
            name="par2", value=10, unit="kg", description="Test parameter 2"
        ),
    ]

    states: List[CasadiState] = [
        CasadiState(name="state", value=290, unit="K", description="Test state")
    ]

    inputs: List[CasadiInput] = [
        CasadiInput(name="myctrl", value=100, unit="W", description="Test control"),
        CasadiInput(
            name="disturbance", value=280, unit="K", description="Test disturbance"
        ),
    ]
    outputs: List[CasadiOutput] = [
        CasadiOutput(name="myout", value=100, description="Test output"),
    ]


class MyCasadiModel(CasadiModel):

    config: MyCasadiModelConfig

    def setup_system(self):
        self.state.ode = (
            self.myctrl + self.par * (self.state - self.disturbance) - self.par2
        )

        self.myout.alg = self.state

        # cost function
        return (self.state - 290) ** 2


# @pytest.fixture
# def example_casadi_model():
#     return MyCasadiModel()
