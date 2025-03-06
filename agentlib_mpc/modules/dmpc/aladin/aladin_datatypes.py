import dataclasses
from itertools import chain
from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd
import scipy
from agentlib.core.module import BaseModuleConfigClass

from agentlib_mpc.data_structures import mpc_datamodels
from agentlib_mpc.data_structures.admm_datatypes import (
    LAG_PREFIX,
    MULTIPLIER_PREFIX,
    LOCAL_PREFIX,
)
from agentlib_mpc.data_structures.coordinator_datatypes import StructuredValue
import agentlib_mpc.data_structures.coordinator_datatypes as cdt


LOCAL_PENALTY_FACTOR = "local_penalty_factor"
PRESCRIBED = "prescribed"


@dataclasses.dataclass
class SparseBuilder:
    values: List[Literal[-1, 1]] = dataclasses.field(default_factory=list)
    row_indices: List[int] = dataclasses.field(default_factory=list)
    column_indices: List[int] = dataclasses.field(default_factory=list)

    def add(self, value: Literal[-1, 1], column: int, row: int):
        """Adds an entry to the sparse matrix"""
        self.values.append(value)
        self.column_indices.append(column)
        self.row_indices.append(row)


@dataclasses.dataclass
class AgentDictEntry(cdt.AgentDictEntry):
    """Holds participating coupling variables (consensus and exchange) of a single
    agent in ADMM. Used in the coordinator."""

    # coup vars contain the mapping to the full local opt vector
    coup_vars: Dict[str, List[int]] = dataclasses.field(default_factory=list)
    coup_vars_parent: List[str] = dataclasses.field(default_factory=list)
    coup_vars_child: List[str] = dataclasses.field(default_factory=list)
    opt_var_length: int = None
    sparse_builder: SparseBuilder = dataclasses.field(default_factory=SparseBuilder)
    coupling_matrix: scipy.sparse.coo_matrix = None
    local_solution: np.ndarray = None
    local_update: np.ndarray = None
    local_target: np.ndarray = None
    hessian: np.ndarray = None
    jacobian: np.ndarray = None
    gradient: np.ndarray = None
    multipliers: Dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def finalize_matrix(self, global_coup_var_length: int):
        b = self.sparse_builder
        self.coupling_matrix = scipy.sparse.coo_matrix(
            (b.values, (b.row_indices, b.column_indices)),
            shape=(global_coup_var_length, self.opt_var_length),
        )


@dataclasses.dataclass
class CouplingEntry:
    """Holds naming conventions for different optimizatin variables / parameters
    associated with a coupling variable in consensus ADMM."""

    name: str

    @property
    def local(self):
        return f"{LOCAL_PREFIX}_{self.name}"

    @property
    def multiplier(self):
        return f"{MULTIPLIER_PREFIX}_{self.name}"

    @property
    def lagged(self):
        return f"{LAG_PREFIX}_{self.name}"

    def aladin_variables(self):
        return [self.local, self.multiplier, self.lagged]


@dataclasses.dataclass
class VariableReference(mpc_datamodels.FullVariableReference):
    """Holds info about all variables of an MPC and their role in the optimization
    problem."""

    couplings: list[CouplingEntry] = dataclasses.field(default_factory=list)

    @classmethod
    def from_config(cls, config: BaseModuleConfigClass):
        """Creates an instance from a pydantic values dict which includes lists of
        AgentVariables with the keys corresponding to 'states', 'inputs', etc.."""
        var_ref: cls = super().from_config(config)
        var_ref.couplings = [CouplingEntry(name=c.name) for c in config.couplings]
        return var_ref

    def all_variables(self) -> List[str]:
        """Returns a list of all variables registered in the var_ref"""
        full_dict = self.__dict__.copy()
        couplings: List[CouplingEntry] = full_dict.pop("couplings")
        coup_vars = []
        for coup in couplings:
            coup_vars.append(coup.name)
        original_vars = list(chain.from_iterable(full_dict.values()))
        return original_vars + coup_vars

    def __contains__(self, item):
        return item in set(self.all_variables())


@dataclasses.dataclass
class Edge:
    """Holds all trajectories belonging to a single edge (i,j) = e of the Eps1
    graph, i.e. a node of the Eps2 graph."""

    parent: np.ndarray
    child: np.ndarray
    edge: np.ndarray
    lower: float
    upper: float
    parent_slack: np.ndarray
    child_slack: np.ndarray
    parent_multiplier: np.ndarray
    child_multiplier: np.ndarray
    parent_lambda: np.ndarray  # slack multiplier
    child_lambda: np.ndarray  # slack multiplier
    _archive: pd.DataFrame = dataclasses.field(default_factory=lambda: pd.DataFrame())

    def to_df(self):
        return pd.DataFrame(self.__dict__)

    def initialize_inner_loop(self, beta: float, calc_multiplier: bool = True):
        """
        Finds x, x_, z, y satisfying lam + beta*z + y = 0

        Args:
            calc_multiplier: If True, adjusts the multiplier to the slack
            beta: Current value of the outer penalty parameter beta.
        Returns:

        """
        if calc_multiplier:
            self.parent_multiplier = -self.parent_lambda - beta * self.parent_slack
            self.child_multiplier = -self.child_lambda - beta * self.child_slack
        else:
            self.parent_slack = (self.parent_lambda - self.parent_multiplier) / beta
            self.child_slack = (self.child_lambda - self.child_multiplier) / beta

    @property
    def unscaled_edge(self) -> np.ndarray:
        """Edge trajectory representing the original physical value without scaling."""
        return self._get_unscaled("edge")

    @unscaled_edge.setter
    def unscaled_edge(self, value: np.ndarray):
        self._set_unscaled("edge", value)

    @property
    def unscaled_parent(self) -> np.ndarray:
        """Parent trajectory representing the original physical value without
        scaling."""
        return self._get_unscaled("parent")

    @unscaled_parent.setter
    def unscaled_parent(self, value: np.ndarray):
        self._set_unscaled("parent", value)

    @property
    def unscaled_child(self) -> np.ndarray:
        """Child trajectory representing the original physical value without scaling."""
        return self._get_unscaled("child")

    @unscaled_child.setter
    def unscaled_child(self, value: np.ndarray):
        self._set_unscaled("child", value)

    def _get_unscaled(self, attr: str):
        # attr can be "edge", "parent", "child"
        scaled = self.__dict__[attr]
        return scaled * (self.upper - self.lower) + self.lower

    def _set_unscaled(self, attr: str, value: np.ndarray):
        # attr can be "edge", "parent", "child"
        value = np.array(value)
        scaled = (value - self.lower) / (self.upper - self.lower)
        self.__dict__[attr] = scaled

    @property
    def trajectories(self) -> List[str]:
        """
        The names of the attributes representing parameter lists, e.g. ['child',
        'parent', 'child_slack', ...
        Used for quick access and looping.
        """
        attrs = list(self.__dict__.keys())
        non_trajectories = ["upper", "lower", "_archive"]
        trajectories = [attr for attr in attrs if attr not in non_trajectories]
        return trajectories

    def shift(self, steps_to_shift: int):
        """
        Shifts all sequences forward by one sampling time.
        Args:
            steps_to_shift:   Number of intervals to be shifted. E.g. 1 for multiple
                shooting, 3 for collocation of third order etc.

        """
        trajectories = self.trajectories
        for name in trajectories:
            values = self.__dict__[name]
            shifted = np.hstack([values[steps_to_shift:], values[-steps_to_shift:]])
            self.__dict__[name] = shifted


@dataclasses.dataclass
class CoordinatorToAgent(StructuredValue):
    target: str
    z: List[float]
    lam: Dict[str, Union[np.ndarray, List[float]]]
    rho: float


@dataclasses.dataclass
class AgentToCoordinator(StructuredValue):
    x: Union[List[float], np.ndarray]  # local solution
    g: Union[List[float], np.ndarray]  # objective gradient
    J: Union[List[List[float]], np.ndarray]  # constraint jacobian
    H: Union[List[List[float]], np.ndarray]  # hessian


@dataclasses.dataclass
class RegistrationA2C(StructuredValue):
    local_solution: np.ndarray
    coup_vars: Dict[str, np.ndarray]
