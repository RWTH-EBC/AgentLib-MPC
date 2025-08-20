"""Holds the classes for CasADi variables and the CasADi model."""

import json
import logging
import abc
from itertools import chain

from typing import List, Union, Tuple, Optional

import attrs
import pandas as pd
from pydantic import Field, PrivateAttr, ConfigDict
import casadi as ca
import numpy as np

from agentlib.core import Model, ModelConfig
from agentlib.core.datamodels import (
    ModelVariable,
    Variability,
    Causality,
)
from agentlib_mpc.data_structures.casadi_utils import ModelConstraint
import warnings

CasadiTypes = Union[ca.MX, ca.SX, ca.DM, ca.Sparsity]

logger = logging.getLogger(__name__)
ca_func_inputs = Union[ca.MX, ca.SX, ca.Sparsity, ca.DM]
ca_all_inputs = Union[ca_func_inputs, np.float64, float]
ca_constraint = Tuple[ca_all_inputs, ca_func_inputs, ca_all_inputs]
ca_constraints = List[Tuple[ca_all_inputs, ca_func_inputs, ca_all_inputs]]


@attrs.define(slots=True, weakref_slot=False, kw_only=True)
class CasadiVariable(ModelVariable):
    """Base Class for variables used in Casadi Models for simulation and
    optimization. Implements the standard arithmetic operations,
    so CasadiVariables can be used in equations.
    Attributes:
        sym: The symbolic CasADi variable used to define ode's and
            optimization problems.
    """

    _sym: CasadiTypes = attrs.field(default=None, alias="_sym")

    def __attrs_post_init__(self):
        self._sym = self.create_sym()

    def create_sym(self) -> ca.MX:
        """Ensures a symbolic MX variable is created with each CasadiVariable
        instance, and that its dimensions are consistent."""
        if self.value is not None:
            if isinstance(self.value, (float, int)):
                shape = (1, 1)
            else:
                shape = np.array(self.value).shape
                if len(shape) == 1:
                    shape = (shape[0], 1)
        else:
            shape = (1, 1)
        sym = ca.MX.sym(self.name, shape[0], shape[1])
        return sym

    @property
    def sym(self) -> ca.MX:
        return self._sym

    def __add__(self, other):
        return self._sym + other

    def __radd__(self, other):
        return other + self._sym

    def __sub__(self, other):
        return self._sym - other

    def __rsub__(self, other):
        return other - self._sym

    def __mul__(self, other):
        return self._sym * other

    def __rmul__(self, other):
        return other * self._sym

    def __truediv__(self, other):
        return self._sym / other

    def __rtruediv__(self, other):
        return other / self._sym

    def __pow__(self, power, modulo=None):
        return self._sym**power

    def __rpow__(self, other):
        return other**self._sym

    def __abs__(self):
        return ca.fabs(self._sym)

    def __matmul__(self, other):
        return self._sym @ other.sym

    def __neg__(self):
        return -self._sym

    def __eq__(self, other):
        try:
            return self.sym == other.sym
        except AttributeError:
            return False

    def __le__(self, other):
        try:
            return self.sym <= other.sym
        except AttributeError as e:
            raise TypeError(
                "Cannot compare a CasadiVariable to a Non-CasadiVariable"
            ) from e

    def __lt__(self, other):
        try:
            return self.sym < other.sym
        except AttributeError as e:
            raise TypeError(
                "Cannot compare a CasadiVariable to a Non-CasadiVariable"
            ) from e

    def __ne__(self, other):
        try:
            return self.sym != other.sym
        except AttributeError:
            return True

    def __ge__(self, other):
        try:
            return self.sym >= other.sym
        except AttributeError as e:
            raise TypeError(
                "Cannot compare a CasadiVariable to a Non-CasadiVariable"
            ) from e

    def __gt__(self, other):
        try:
            return self.sym > other.sym
        except AttributeError as e:
            raise TypeError(
                "Cannot compare a CasadiVariable to a Non-CasadiVariable"
            ) from e


@attrs.define(slots=True, weakref_slot=False, kw_only=True)
class CasadiParameter(CasadiVariable):
    """
    Class that stores various attributes of parameters.
    """

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.causality: Causality = Causality.parameter
        self.variability: Variability = Variability.tunable


@attrs.define(slots=True, weakref_slot=False, kw_only=True)
class CasadiState(CasadiVariable):
    """
    Class that stores various attributes of CasADi differential variables.
    """

    _ode: Optional[CasadiTypes] = attrs.field(default=None, alias="_ode")

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.causality: Causality = Causality.local
        self.variability: Variability = Variability.continuous

    @property
    def alg(self) -> CasadiTypes:
        raise AttributeError(
            "Casadi States should not have .alg assignments. If you wish to provide "
            "algebraic relationships to states, add them in the constraints."
        )
        return -1

    @alg.setter
    def alg(self, equation: Union[CasadiTypes, CasadiVariable]):
        raise AttributeError(
            "Casadi States should not have .alg assignments. Consider the following: \n"
            " 1. If you need equality constraints in your MPC, please add them in the "
            "constraints. \n"
            " 2. If you use this to bundle an expression, consider using a regular "
            "Python variable. \n"
            " 3. Implicit algebraic equations are currently not supported."
        )

    @property
    def ode(self) -> CasadiTypes:
        return self._ode

    @ode.setter
    def ode(self, equation: Union[CasadiTypes, CasadiVariable]):
        self._ode = get_symbolic(equation)

    def json(self, indent: int = 2, **kwargs):
        data = self.dict(**kwargs)
        if isinstance(self.value, pd.Series):
            data["value"] = self.value.to_dict()
        data.pop("_ode")
        data.pop("_alg")
        json.dumps(data, indent=indent)


@attrs.define(slots=True, weakref_slot=False, kw_only=True)
class CasadiInput(CasadiVariable):
    """
    Class that stores various attributes of control variables.
    """

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.causality: Causality = Causality.input
        self.variability: Variability = Variability.continuous

    @property
    def alg(self) -> CasadiTypes:
        raise AttributeError(
            "Casadi Inputs should not have .alg assignments. If you wish to provide "
            "algebraic relationships to states, add them in the constraints."
        )
        return -1

    @alg.setter
    def alg(self, equation: Union[CasadiTypes, CasadiVariable]):
        raise ValueError(
            "Cannot assign algebraic equations to inputs. If this is for an MPC, "
            "try defining a constraint instead."
        )


@attrs.define(slots=True, weakref_slot=False, kw_only=True)
class CasadiOutput(CasadiVariable):
    """
    Class that stores various attributes of control variables.
    """

    _alg: CasadiTypes = attrs.field(default=None, alias="_alg")

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.causality: Causality = Causality.output
        self.variability: Variability = Variability.continuous

    @property
    def alg(self) -> CasadiTypes:
        return self._alg

    @alg.setter
    def alg(self, equation: Union[CasadiTypes, CasadiVariable]):
        if isinstance(equation, CasadiVariable):
            # Converts CasadiVariables to their symbolic variable. Useful in case
            # CasadiVariables are assigned in equations as is, i.e. their math methods
            # are not called.
            self._alg = equation.sym
        else:
            self._alg = equation

    def json(self, **kwargs):
        data = self.dict(**kwargs)
        if isinstance(self.value, pd.Series):
            data["value"] = self.value.to_dict()
        data.pop("_alg")
        json.dumps(data)


class CasadiModelConfig(ModelConfig):
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


class CasadiModel(Model):
    """Base Class for CasADi models. To implement your own model, inherit
    from this class, specify the variables (inputs, outputs, states,
    parameters and override the setup_system() method."""

    config: CasadiModelConfig

    def __init__(self, **kwargs):
        # Temporarily disable __setattr__ checks during initialization as self.config does not exists at first
        super().__setattr__("_is_initialized", False)
        # Initializes the config
        super().__init__(**kwargs)
        self._is_initialized = True
        # Check forbidden names
        bad_variable_names = self._get_forbidden_variable_names().intersection(
            self.config.get_variable_names()
        )
        if bad_variable_names:
            raise NameError(
                "The following variable names are not allowed as the intersect "
                f"with internal names of {self.__class__.__name__}: "
                f"{' ,'.join(bad_variable_names)}"
            )

        self.constraints = []  # constraint functions
        self.time = ca.MX.sym("time", 1, 1)

        # read constraints, assign ode's and return cost function
        objective_result = self.setup_system()
        self._assert_outputs_are_defined()

        # Handle both new and old objective formats
        if hasattr(objective_result, "get_casadi_expression"):
            # New objective system
            self.objective = objective_result
            self.cost_func = objective_result.get_casadi_expression()
        else:
            # Old objective system - backwards compatibility
            warnings.warn(
                "\033[93mWARNING:\033[0m Model uses the deprecated objective formulation. "
                "Consider migrating to the new FullObjective formulation.\n")
            self.cost_func = objective_result
            self.objective = None

        # save system equations as a single casadi vector
        system = ca.vertcat(*[sta.ode for sta in self.differentials])
        # prevents errors in case system is empty
        self.system = ca.reshape(system, system.shape[0], 1)
        self.integrator = None  # set in intitialize
        self.initialize()

    @staticmethod
    def _get_forbidden_variable_names() -> set[str]:
        """
        Function gives all variable names which are forbidden
        due to the fact that we override __setattr__ in order
        to avoid users creating instance attributes for variable
        names.
        If a user names a variable, e.g. "constraints", the
        error would not point to the variable name being a
        bad choice.

        Returns:
            Set of forbidden names as str
        """
        return {
            "constraints",
            "cost_func",
            "time",
            "system",
            "integrator",
        }

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
        pars = self.get_input_values(t_start)
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
            *[inp.sym for inp in chain.from_iterable([self.inputs, self.parameters])], self.time
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

    def get_input_values(self, t_start):
        return ca.vertcat(
            *[inp.value for inp in chain.from_iterable([self.inputs, self.parameters])],t_start
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
        if self._is_initialized and key in self.config.get_variable_names():
            raise AttributeError(
                f"You are trying to create an instance attribute with the name {key}, "
                f"which is also in the variables of {self.__class__.__name__}. This can "
                f"lead to unwanted behaviour. Assign variables or equations via .alg "
                f"for CasadiOutputs and .ode for CasadiStates. To change other variable attributes, "
                f"use .value, .lb or similar."
            )
        # todo


def get_symbolic(equation):
    if isinstance(equation, CasadiVariable):
        # Converts CasadiVariables to their symbolic variable. Useful in case
        # CasadiVariables are assigned in equations as is, i.e. their math methods
        # are not called.
        return equation.sym
    else:
        return equation
