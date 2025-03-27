import dataclasses
from pathlib import Path
from typing import List, Union, TypeVar, Protocol, Sequence, Iterable, Optional
from itertools import chain

import attrs
import numpy as np
import pandas as pd
import pydantic
from enum import Enum, auto
from agentlib.core import AgentVariable
from agentlib.core.module import BaseModuleConfigClass

from agentlib_mpc.data_structures.interpolation import InterpolationMethods
from pydantic import ConfigDict


class InitStatus(str, Enum):
    """Keep track of the readyness status of the MPC."""

    pre_module_init = auto()
    during_update = auto()
    ready = auto()


class DiscretizationOptions(pydantic.BaseModel):
    """Class defining the options to discretize an MPC. Can be extended for different
    optimization implementations."""

    model_config = ConfigDict(extra="allow")

    time_step: float = pydantic.Field(
        default=60,
        ge=0,
        description="Time step of the MPC.",
    )
    prediction_horizon: int = pydantic.Field(
        default=5,
        ge=0,
        description="Prediction horizon of the MPC.",
    )


class Results(Protocol):
    df: pd.DataFrame

    def __getitem__(self, item: str) -> Sequence[float]: ...


@dataclasses.dataclass
class BaseVariableReference:
    def all_variables(self) -> List[str]:
        """Returns a list of all variables registered in the var_ref"""
        return list(chain.from_iterable(self.__dict__.values()))

    @classmethod
    def from_config(cls, config: BaseModuleConfigClass):
        """Creates an instance from a pydantic values dict which includes lists of
        AgentVariables with the keys corresponding to 'states', 'inputs', etc.."""

        def names_list(ls: List[AgentVariable]):
            return [item.name for item in ls]

        field_names = set(f.name for f in dataclasses.fields(cls))
        variables = {
            k: names_list(v) for k, v in config.__dict__.items() if k in field_names
        }
        return cls(**variables)

    def __contains__(self, item):
        all_variables = set(chain.from_iterable(self.__dict__.values()))
        return item in all_variables

VariableReferenceT = TypeVar("VariableReferenceT", bound=BaseVariableReference)


@dataclasses.dataclass
class VariableReference(BaseVariableReference):
    states: List[str] = dataclasses.field(default_factory=list)
    controls: List[str] = dataclasses.field(default_factory=list)
    inputs: List[str] = dataclasses.field(default_factory=list)
    parameters: List[str] = dataclasses.field(default_factory=list)
    outputs: List[str] = dataclasses.field(default_factory=list)


def r_del_u_convention(name: str) -> str:
    """Turns the name of a control variable into its weight via convention"""
    return f"r_del_u_{name}"


@dataclasses.dataclass
class FullVariableReference(VariableReference):
    @property
    def r_del_u(self) -> List[str]:
        return [r_del_u_convention(cont) for cont in self.controls]


@dataclasses.dataclass
class MINLPVariableReference(VariableReference):
    binary_controls: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class MHEVariableReference(BaseVariableReference):
    states: List[str] = dataclasses.field(default_factory=list)
    measured_states: List[str] = dataclasses.field(default_factory=list)
    weights_states: List[str] = dataclasses.field(default_factory=list)
    estimated_inputs: List[str] = dataclasses.field(default_factory=list)
    estimated_parameters: List[str] = dataclasses.field(default_factory=list)
    known_inputs: List[str] = dataclasses.field(default_factory=list)
    known_parameters: List[str] = dataclasses.field(default_factory=list)
    outputs: List[str] = dataclasses.field(default_factory=list)

    def all_variables(self) -> Iterable[str]:
        """Returns a list of all variables registered in the var_ref which the MHE can
        get from the config with get()"""
        return (
            set(super().all_variables())
            - set(self.measured_states)
            - set(self.weights_states)
        )


@attrs.define(slots=True, weakref_slot=False, kw_only=True)
class MPCVariable(AgentVariable):
    """AgentVariable used to define input variables of MPC."""

    interpolation_method: InterpolationMethods = attrs.field(
        default=InterpolationMethods.linear,
        metadata={
            "description": "Defines which method is used for interpolation of "
            "boundaries or  values for this variable. Default is linear.",
            "title": "Interpolation Method",
        },
    )


@attrs.define(slots=True, weakref_slot=False, kw_only=True)
class MPCSeriesVariable(MPCVariable):
    """MPCVariable with pd.Series as the default type."""

    type: Optional[str] = attrs.field(
        default=None,
        metadata={
            "title": "Type",
            "description": "Name the type of the variable using a string",
        },
    )


MPCVariables = List[MPCVariable]
MPCSeriesVariables = List[MPCSeriesVariable]


def stats_path(path: Union[Path, str]) -> Path:
    res_file = Path(path)
    return Path(res_file.parent, "stats_" + res_file.name)


def cia_relaxed_results_path(path: Union[Path, str]) -> Path:
    res_file = Path(path)
    return Path(res_file.parent, "relaxed_" + res_file.name)


MPCValue = Union[int, float, list[Union[int, float]], pd.Series, np.ndarray]
