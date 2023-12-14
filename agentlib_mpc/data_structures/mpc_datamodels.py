import abc
import dataclasses
from pathlib import Path
from typing import List, Union, TypeVar, Protocol
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


@dataclasses.dataclass
class VariableReference:
    states: List[str] = dataclasses.field(default_factory=list)
    controls: List[str] = dataclasses.field(default_factory=list)
    inputs: List[str] = dataclasses.field(default_factory=list)
    parameters: List[str] = dataclasses.field(default_factory=list)
    outputs: List[str] = dataclasses.field(default_factory=list)

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


VariableReferenceT = TypeVar("VariableReferenceT", bound=VariableReference)


def r_del_u_convention(name: str) -> str:
    """Turns the name of a control variable into its weight via convention"""
    return f"r_del_u_{name}"


@dataclasses.dataclass
class FullVariableReference(VariableReference):
    @property
    def r_del_u(self) -> List[str]:
        return [r_del_u_convention(cont) for cont in self.controls]


@dataclasses.dataclass
class MIQPVariableReference(VariableReference):
    binary_controls: List[str] = dataclasses.field(default_factory=list)


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


MPCVariables = List[MPCVariable]

#####################################################################
#                       Soft Constraints                            #
#####################################################################


class SoftConstraintPrefix(str, Enum):
    slack = "slack"
    weight = "weight"
    value = "value"


def soft_constraint_naming(var: AgentVariable, *, prefix: str) -> str:
    """Returns a variable name for generated variables for soft constraints."""
    return f"{prefix}_{var.alias}_{var.source}"


def sc_slack_name(var) -> str:
    return soft_constraint_naming(var, prefix=SoftConstraintPrefix.slack)


def sc_weight_name(var) -> str:
    return soft_constraint_naming(var, prefix=SoftConstraintPrefix.weight)


def sc_value_name(var) -> str:
    return soft_constraint_naming(var, prefix=SoftConstraintPrefix.value)


def stats_path(path: Union[Path, str]) -> Path:
    res_file = Path(path)
    return Path(res_file.parent, "stats_" + res_file.name)


MPCValue = Union[int, float, list[Union[int, float]], pd.Series, np.ndarray]


class Results(abc.ABC):
    """Specifies the optimization results. Should be returned from the backend to the
    mpc. Used in the mpc for further processing and inter agent communication, and for
    saving. Holds the discretized full results over the prediction horizon, as well as
    the solve stats."""
    columns: pd.MultiIndex
    stats: dict
    variable_grid_indices: dict[str, list[int]]
    _variable_name_to_index: dict[str, int] = None

    def __getitem__(self, item: str) -> np.ndarray:
        return self.df()[item]

    @abc.abstractmethod
    def df(self) -> pd.DataFrame:
        raise NotImplementedError

    def write_columns(self, file: Path):
        """Write an empty results file with the correct columns."""
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(file)

    def write_stats_columns(self, file: Path):
        """Write an empty stats file with the correct columns."""
        line = f""",{",".join(self.stats)}\n"""
        with open(file, "w") as f:
            f.write(line)

    def stats_line(self, index: str) -> str:
        """Create a line, that should be appended to the stats file."""
        return f""""{index}",{",".join(map(str, self.stats.values()))}\n"""
