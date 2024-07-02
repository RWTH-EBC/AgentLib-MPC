from pygrampc import ProblemBase
from typing import List
import logging

from agentlib_mpc.models.grampc_model import GrampcModel, GrampcModelConfig
from agentlib.core import ModelInput, ModelOutput, ModelState, ModelParameter

logger = logging.getLogger(__name__)


class OneZoneModelConfig(GrampcModelConfig):
    inputs: List[ModelInput] = [
        ModelInput(name="mDot", value=0.0, unit="m³/s", description="Air mass flow into zone"),
        ModelInput(name="load", value=150.0, unit="W", description="Heat load into zone"),
        ModelInput(name="T_in", value=17, unit="°C", description="Inflow air temperature"),
    ]

    states: List[ModelState] = [
        ModelState(name="T", value=25.0, unit="°C", description="Zone temperature")
    ]

    outputs: List[ModelOutput] = [
        ModelOutput(name="T_out", value=0.0, unit="°C", description="Zone temperature")
    ]

    parameters: List[ModelParameter] = [
        ModelParameter(name="cp", value=1000, unit="J/kg*K", description="thermal capacity of the air", ),
        ModelParameter(name="C", value=100000, unit="J/K", description="thermal capacity of zone"),
    ]


class OneZoneModel(GrampcModel):
    config: OneZoneModelConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        parameters = self.get_parameters(["C", "cp"])
        inputs = self.get_inputs(["T_in", "load"])
        self.grampc_problem = OneZone(*parameters, *inputs)

    def output_function(self, time, states, controls, parameters):
        return states


class OneZone(ProblemBase):
    def __init__(self, C, cp, Tin, load):
        ProblemBase.__init__(self)
        # Problem dimensions for GRAMPC
        self.Nx = 1
        self.Nu = 1
        self.Np = 0
        self.Ng = 0
        self.Nh = 2
        self.NgT = 0
        self.NhT = 0

        # parameters
        self.C = C
        self.cp = cp
        self.Tin = Tin

        # disturbances
        self.load = load

        # weights
        self.R = 1.0

    def ffct(self, out, t, x, u, p):
        out[0] = (self.cp * u[0] * (self.Tin - x[0]) + self.load)/self.C

    def dfdx_vec(self, out, t, x, vec, u, p):
        out[0] = -(self.cp * u[0] * x[0]) / self.C * vec[0]

    def dfdu_vec(self, out, t, x, vec, u, p):
        out[0] = self.cp * (self.Tin - x[0]) / self.C * vec[0]

    def lfct(self, out, t, x, u, p, xdes, udes):
        out[0] = self.R * (u[0] - udes[0])**2

    def dldu(self, out, t, x, u, p, xdes, udes):
        out[0] = 2 * self.R * (u[0] - udes[0])

    def hfct(self, out, t, x, u, p):
        out[0] = x[0] - 22.0  # upper bound
        out[1] = 15 - x[0]  # lower bound

    def dhdx_vec(self, out, t, x, u, p, vec):
        out[0] = vec[0] - vec[1]
