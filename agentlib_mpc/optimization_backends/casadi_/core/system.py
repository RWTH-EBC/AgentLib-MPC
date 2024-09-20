"""Holds the System class, which knows the model"""

from __future__ import annotations

import abc
from typing import TypeVar, Union

from agentlib_mpc.data_structures.mpc_datamodels import VariableReference
from agentlib_mpc.models.casadi_model import CasadiModel
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationVariable,
    OptimizationParameter,
)


class System(abc.ABC):
    """

    Examples:
        class MySystem(System):

            # variables
            states: OptimizationVariable
            controls: OptimizationVariable
            algebraics: OptimizationVariable
            outputs: OptimizationVariable

            # parameters
            non_controlled_inputs: OptimizationParameter
            model_parameters: OptimizationParameter
            initial_state: OptimizationParameter

            # dynamics
            model_constraints: Constraint
            cost_function: ca.MX
            ode: ca.MX

            def initialize(self, model: CasadiModel, var_ref: VariableReference):

                self.states = OptimizationVariable.declare(
                    denotation="state",
                    variables=model.get_states(var_ref.states),
                    ref_list=var_ref.states,
                    assert_complete=True,
                )

                .
                .
                .
        )
    """

    @abc.abstractmethod
    def initialize(self, model: CasadiModel, var_ref: VariableReference): ...

    @property
    def variables(self) -> list[OptimizationVariable]:
        return [
            var
            for var in self.__dict__.values()
            if isinstance(var, OptimizationVariable)
        ]

    @property
    def parameters(self) -> list[OptimizationParameter]:
        return [
            var
            for var in self.__dict__.values()
            if isinstance(var, OptimizationParameter)
        ]

    @property
    def quantities(self) -> list[Union[OptimizationParameter, OptimizationVariable]]:
        return self.variables + self.parameters


SystemT = TypeVar("SystemT", bound=System)
