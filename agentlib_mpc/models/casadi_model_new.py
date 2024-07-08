"""Holds the classes for CasADi variables and the CasADi model."""
from __future__ import annotations

import logging
import abc
from itertools import chain

from typing import List, Union, Tuple, Optional

import attrs
from pydantic import Field, PrivateAttr, ConfigDict
import casadi as ca
import numpy as np

from agentlib.core import Model, ModelConfig
from agentlib.core.datamodels import (
    Causality,
)
from agentlib_mpc.data_structures.casadi_utils import ModelConstraint
from agentlib_mpc.models.casadi_model import CasadiInput, CasadiOutput, CasadiState, \
    CasadiParameter, CasadiVariable

CasadiTypes = Union[ca.MX, ca.SX, ca.DM, ca.Sparsity]

logger = logging.getLogger(__name__)
ca_func_inputs = Union[ca.MX, ca.SX, ca.Sparsity, ca.DM]
ca_all_inputs = Union[ca_func_inputs, np.float64, float]
ca_constraint = Tuple[ca_all_inputs, ca_func_inputs, ca_all_inputs]
ca_constraints = List[Tuple[ca_all_inputs, ca_func_inputs, ca_all_inputs]]



@attrs.define(slots=True, weakref_slot=False, kw_only=True)
class Var(CasadiVariable):
    name: str = "__empty__"
    causality: Causality
    @classmethod
    def input(cls, **kwargs) -> Var:
        return cls(**kwargs, causality=Causality.input)

    @classmethod
    def parameter(cls, **kwargs) -> Var:
        return cls(**kwargs, causality=Causality.parameter)

    @classmethod
    def state(cls, **kwargs) -> Var:
        return cls(**kwargs, causality=Causality.local)

    @classmethod
    def output(cls, **kwargs) -> Var:
        return cls(**kwargs, causality=Causality.output)



        #
        #
        # name: str = field(
        #     metadata={"title": "Name", "description": "The name of the variable"}
        # )
        # type: #Optional[str] = field(
        #     default=None,
        #     metadata={
        #         "title": "Type",
        #         "description": "Name the type of the variable using a string",
        #     },
        # )
        # timestamp: Optional[float] = field(
        #     default=None,
        #     metadata={
        #         "title": "Timestamp",
        #         "description": "Timestamp of the current value",
        #     },
        # )
        # unit: str = field(
        # description: str
        # ub: Union[float, int] = field(
        #     default=math.inf,
        # lb: Union[float, int] = field(
        #     default=-math.inf,
        # clip: bool
        # value: Any


class CasadiModelConfig2(ModelConfig):
    system: CasadiTypes = None
    cost_function: CasadiTypes = None

    inputs: List[CasadiInput] = Field(default=list())
    outputs: List[CasadiOutput] = Field(default=list())
    states: List[CasadiState] = Field(default=list())
    parameters: List[CasadiParameter] = Field(default=list())
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    _types: dict[str, type] = PrivateAttr(
        default={
            "inputs": CasadiInput,
            "outputs": CasadiOutput,
            "states": CasadiState,
            "parameters": CasadiParameter,
        }
    )



class CasadiModel2(Model):
    """Base Class for CasADi models. To implement your own model, inherit
    from this class, specify the variables (inputs, outputs, states,
    parameters and override the setup_system() method."""

    config: CasadiModelConfig2

    def __init__(self, **kwargs):
        # Initializes the config
        super().__init__(**kwargs)

        self.constraints = []  # constraint functions
        # read constraints, assign ode's and return cost function
        self.cost_func = self.setup_system()
        self._assert_outputs_are_defined()

        # save system equations as a single casadi vector
        system = ca.vertcat(*[sta.ode for sta in self.differentials])
        # prevents errors in case system is empty
        self.system = ca.reshape(system, system.shape[0], 1)
        self.integrator = None  # set in intitialize
        self.initialize()

    def _assert_outputs_are_defined(self):
        """Raises an Error, if the output variables are not defined with an equation"""
        for out in self.outputs:
            if out.alg is None:
                raise ValueError(
                    f"Output '{out.name}' was not initialized with an equation. Make "
                    f"sure you specify '{out.name}.alg' in 'setup_system()'."
                )

    def do_step(self, *, t_start, t_sample=None):
        if t_sample is None:
            t_sample = self.dt
        pars = self.get_input_values()
        t_sim = 0
        if self.differentials:
            x0 = self.get_differential_values()
            curr_x = x0
            while t_sim < t_sample:
                result = self.integrator(x0=curr_x, p=pars)
                t_sim += self.dt
                curr_x = result["xf"]
            self.set_differential_values(np.array(result["xf"]).flatten())
        else:
            result = self.integrator(p=pars)
        if self.outputs:
            self.set_output_values(np.array(result["zf"]).flatten())

    def _make_integrator(self) -> ca.Function:
        """Creates the integrator to be used in do_step(). The integrator takes the
        current state and input values as input and returns the state values and
        algebraic values at the end of the interval."""
        opts = {"t0": 0, "tf": self.dt}
        par = ca.vertcat(
            *[inp.sym for inp in chain.from_iterable([self.inputs, self.parameters])]
        )
        x = ca.vertcat(*[sta.sym for sta in self.differentials])
        z = ca.vertcat(*[var.sym for var in self.outputs])
        algebraic_equations = ca.vertcat(*self.output_equations)

        if not algebraic_equations.shape[0] and self.differentials:
            # case of pure ode
            ode = {"x": x, "p": par, "ode": self.system}
            integrator = ca.integrator("system", "cvodes", ode, opts)
        elif algebraic_equations.shape[0] and self.differentials:
            # mixed dae
            dae = {
                "x": x,
                "p": par,
                "ode": self.system,
                "z": z,
                "alg": algebraic_equations,
            }
            integrator = ca.integrator("system", "idas", dae, opts)

        else:
            # only algebraic equations
            dae = {
                "x": ca.MX.sym("dummy", 1),
                "p": par,
                "ode": 0,
                "z": z,
                "alg": algebraic_equations,
            }
            integrator_ = ca.integrator("system", "idas", dae, opts)
            integrator = ca.Function(
                "system", [par], [integrator_(x0=0, p=par)["zf"]], ["p"], ["zf"]
            )
        return integrator

    def initialize(self, **ignored):
        """
        Initializes Casadi model. Creates the integrator to be used in
        do_step(). The integrator takes the current state and input values as
        input and returns the state values at the end of the interval and the
        value of the cost function integrated over the interval.
        """
        self.integrator = self._make_integrator()

    def get_constraints(self) -> List[ModelConstraint]:
        """List of constraints of the form (lower, function, upper)."""
        base_constraints = [
            ModelConstraint(lb * 1, func * 1, ub * 1)
            for lb, func, ub in self.constraints
        ]
        equality_constraints = [
            ModelConstraint(0, alg, 0) for alg in self.output_equations
        ]
        return base_constraints + equality_constraints

    @property
    def inputs(self) -> list[CasadiInput]:
        """Get all model inputs as a list"""
        return list(self._inputs.values())

    @property
    def outputs(self) -> list[CasadiOutput]:
        """Get all model outputs as a list"""
        return list(self._outputs.values())

    @property
    def states(self) -> list[CasadiState]:
        """Get all model states as a list"""
        return list(self._states.values())

    @property
    def parameters(self) -> list[CasadiParameter]:
        """Get all model parameters as a list"""
        return list(self._parameters.values())

    @property
    def output_equations(self) -> List[CasadiTypes]:
        """List of algebraic equations RHS in the form
        0 = z - g(x, z, p, ... )"""
        return [alg_var - alg_var.alg for alg_var in self.outputs]

    @property
    def differentials(self) -> List[CasadiState]:
        """List of all CasadiStates with an associated differential equation."""
        return [var for var in self.states if var.ode is not None]

    @property
    def auxiliaries(self) -> List[CasadiState]:
        """List of all CasadiStates without an associated equation. Common
        uses for this are slack variables that appear in cost functions and
        constraints of optimization models."""
        return [var for var in self.states if var.ode is None]

    @abc.abstractmethod
    def setup_system(self):
        raise NotImplementedError(
            "The ode is defined by the actual models " "inheriting from this class."
        )

    def get_input_values(self):
        return ca.vertcat(
            *[inp.value for inp in chain.from_iterable([self.inputs, self.parameters])]
        )

    def get_differential_values(self):
        return ca.vertcat(*[sta.value for sta in self.differentials])

    def set_differential_values(self, values: Union[List, np.ndarray]):
        """Sets the values for all differential variables. Provided values list MUST
        match the order in which differentials are saved, there is no check."""
        for state, value in zip(self.differentials, values):
            self._states[state.name].value = value

    def set_output_values(self, values: Union[List, np.ndarray]):
        """Sets the values for all outputs. Provided values list MUST match the order
        in which outputs are saved, there is no check."""
        for var, value in zip(self.outputs, values):
            self._outputs[var.name].value = value

    def get(self, name: str) -> CasadiVariable:
        return super().get(name)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        # todo


def get_symbolic(equation):
    if isinstance(equation, CasadiVariable):
        # Converts CasadiVariables to their symbolic variable. Useful in case
        # CasadiVariables are assigned in equations as is, i.e. their math methods
        # are not called.
        return equation.sym
    else:
        return equation


if __name__ == '__main__':
    a = Var(value=10, unit="asdf")
    print(a)