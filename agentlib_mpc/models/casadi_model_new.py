"""Holds the classes for CasADi variables and the CasADi model."""

from __future__ import annotations

import logging
import abc
from itertools import chain

from typing import List, Union, Tuple, Type, Dict, Any, get_type_hints

import attrs
import pydantic
from pydantic import Field, PrivateAttr, ConfigDict, model_validator
import casadi as ca
import numpy as np

from agentlib.core import Model, ModelConfig
from agentlib.core.datamodels import (
    Causality,
)
from agentlib_mpc.data_structures.casadi_utils import ModelConstraint
from agentlib_mpc.models.casadi_model import (
    CasadiInput,
    CasadiOutput,
    CasadiState,
    CasadiParameter,
    CasadiVariable,
)

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

    def to_casadi_variable(
        self, name: str
    ) -> Union[CasadiInput, CasadiOutput, CasadiState, CasadiParameter]:
        casadi_type = {
            Causality.input: CasadiInput,
            Causality.output: CasadiOutput,
            Causality.local: CasadiState,
            Causality.parameter: CasadiParameter,
        }[self.causality]

        keys_to_replace = {'ode': '_ode', 'alg': '_alg'}
        # dict_ =  {keys_to_replace.get(k, k): v for k, v in self.dict().items()}
        cas_var = casadi_type(**self.dict())
        if hasattr(self, "ode"):
            cas_var.ode = self.ode
        if hasattr(self, "alg"):
            cas_var.alg = self.alg

        return cas_var

    def set_name(self, name):
        if self.name != "__empty__" and name != self.name:
            raise ValueError(
                "Defined a Var with mismatching name. Name needs to match between "
                f"variable definition and class attribute. Var name is {self.name} "
                f"but class attribute was {name}."
            )
        self.name = name
        self._sym = self.create_sym()


class CasadiModelConfig2(ModelConfig, abc.ABC):
    system: CasadiTypes = None
    cost_function: ca_all_inputs = None
    constraints: List[Tuple[Any, Any, Any]] = None

    inputs: List[CasadiInput] = Field(default_factory=list)
    outputs: List[CasadiOutput] = Field(default_factory=list)
    states: List[CasadiState] = Field(default_factory=list)
    parameters: List[CasadiParameter] = Field(default_factory=list)

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
        ignored_types=(Var, ca_all_inputs, CasadiTypes),
    )
    _types: Dict[str, Type] = PrivateAttr(
        default={
            "inputs": CasadiInput,
            "outputs": CasadiOutput,
            "states": CasadiState,
            "parameters": CasadiParameter,
        }
    )
    _started_setup: bool = PrivateAttr(default=False)

    @abc.abstractmethod
    def setup_system(self) -> ca_all_inputs:
        pass

    @model_validator(mode="after")
    def setup_and_collect_variables(self) -> "CasadiModelConfig2":

        # this check makes sure that we only validate once, as this validator performs
        # assignments that would be validated.
        if self._started_setup:
            return self
        self._started_setup = True

        # Get all extra attributes of the class
        all_attrs = set(dir(self.__class__))
        custom_attrs = all_attrs - set(dir(CasadiModelConfig2))

        for field_name in custom_attrs:
            value = getattr(self, field_name, None)
            if not isinstance(value, Var):
                raise AttributeError(
                    f"Extra Field '{field_name}' is not allowed. Add variables only "
                    f"of the Var type defined in {__name__}."
                )
            value.set_name(field_name)

        self.setup_system()

        casadi_vars = {
            Causality.input: self.inputs,
            Causality.output: self.outputs,
            Causality.local: self.states,
            Causality.parameter: self.parameters,
        }



        for field_name in custom_attrs:
            value = getattr(self, field_name)
            casadi_var = value.to_casadi_variable(field_name)
            casadi_vars[value.causality].append(casadi_var)
            # delattr(self, field_name)

        return self


class CasadiModel2(Model):
    """Base Class for CasADi models. To implement your own model, inherit
    from this class, specify the variables (inputs, outputs, states,
    parameters and override the setup_system() method."""

    config: CasadiModelConfig2

    def __init__(self, **kwargs):
        # Initializes the config
        super().__init__(**kwargs)

        self._assert_outputs_are_defined()

        # save system equations as a single casadi vector
        system = ca.vertcat(*[sta.ode for sta in self.differentials])
        # prevents errors in case system is empty
        self.system = ca.reshape(system, system.shape[0], 1)
        self.integrator = None  # set in intitialize
        self.initialize()

    @property
    def cost_func(self) -> ca_all_inputs:
        return self.config.cost_function

    @property
    def constraints(self) -> ca_constraints:
        return self.config.constraints

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


if __name__ == "__main__":
    a = Var(value=10, unit="asdf")
    print(a)
