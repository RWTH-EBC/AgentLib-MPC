import dataclasses
from typing import List, Dict, Iterable, Tuple
from itertools import chain

import numpy as np

import agentlib as al
import orjson
from agentlib.core.module import BaseModuleConfigClass

import agentlib_mpc.data_structures.coordinator_datatypes as cdt
from agentlib_mpc.data_structures import mpc_datamodels

############################## Uncoordinated ADMM ##################################

ADMM_PREFIX = "admm"
MULTIPLIER_PREFIX = ADMM_PREFIX + "_lambda"
LOCAL_PREFIX = ADMM_PREFIX + "_coupling"
MEAN_PREFIX = ADMM_PREFIX + "_coupling_mean"
LAG_PREFIX = ADMM_PREFIX + "_lag"
EXCHANGE_MULTIPLIER_PREFIX = ADMM_PREFIX + "_exchange_lambda"
EXCHANGE_LOCAL_PREFIX = ADMM_PREFIX + "_exchange"
EXCHANGE_MEAN_PREFIX = ADMM_PREFIX + "_exchange_mean"


@dataclasses.dataclass
class CouplingEntry:
    """Holds naming conventions for different optimizatin variables / parameters
    associated with a coupling variable in consensus ADMM."""

    name: str

    @property
    def local(self):
        return f"{LOCAL_PREFIX}_{self.name}"

    @property
    def mean(self):
        return f"{MEAN_PREFIX}_{self.name}"

    @property
    def multiplier(self):
        return f"{MULTIPLIER_PREFIX}_{self.name}"

    @property
    def lagged(self):
        return f"{LAG_PREFIX}_{self.name}"

    def admm_variables(self):
        return [self.local, self.mean, self.multiplier, self.lagged]


@dataclasses.dataclass
class ExchangeEntry:
    """Holds naming conventions for different optimizatin variables / parameters
    associated with a coupling variable in exchange ADMM."""

    name: str

    @property
    def local(self):
        return f"{EXCHANGE_LOCAL_PREFIX}_{self.name}"

    @property
    def mean_diff(self):
        return f"{EXCHANGE_MEAN_PREFIX}_{self.name}"

    @property
    def multiplier(self):
        return f"{EXCHANGE_MULTIPLIER_PREFIX}_{self.name}"

    @property
    def lagged(self):
        return f"{LAG_PREFIX}_{self.name}"

    def admm_variables(self):
        return [self.local, self.mean_diff, self.multiplier, self.lagged]


@dataclasses.dataclass
class VariableReference(mpc_datamodels.FullVariableReference):
    """Holds info about all variables of an MPC and their role in the optimization
    problem."""

    couplings: list[CouplingEntry] = dataclasses.field(default_factory=list)
    exchange: list[ExchangeEntry] = dataclasses.field(default_factory=list)

    @classmethod
    def from_config(cls, config: BaseModuleConfigClass):
        """Creates an instance from a pydantic values dict which includes lists of
        AgentVariables with the keys corresponding to 'states', 'inputs', etc.."""
        var_ref: cls = super().from_config(config)
        var_ref.couplings = [CouplingEntry(name=c.name) for c in config.couplings]
        var_ref.exchange = [ExchangeEntry(name=c.name) for c in config.exchange]
        return var_ref

    def all_variables(self) -> List[str]:
        """Returns a list of all variables registered in the var_ref"""
        full_dict = self.__dict__.copy()
        couplings: List[CouplingEntry] = full_dict.pop("couplings")
        exchange: List[CouplingEntry] = full_dict.pop("exchange")
        coup_vars = []
        for coup in couplings + exchange:
            coup_vars.append(coup.name)
        original_vars = list(chain.from_iterable(full_dict.values()))
        return original_vars + coup_vars

    def __contains__(self, item):
        return item in set(self.all_variables())


def coupling_alias(alias: str) -> str:
    """Naming convention for local variables to send and receive."""
    return f"{LOCAL_PREFIX}_{alias}"


def exchange_alias(alias: str) -> str:
    """Naming convention for local exchange variables to send and receive."""
    return f"{EXCHANGE_LOCAL_PREFIX}_{alias}"


############################## Coordinated ADMM ##################################

# ALIASES
ADMM_COMMUNICATION = "admm_communication"
ADMM_SIGNUP_REQUEST = "admm_signup_request"


glob_params: List[str] = ["penalty_factor", "prediction_horizon", "time_step"]

PENALTY_FACTOR = "penalty_factor"


@dataclasses.dataclass
class AgentDictEntry(cdt.AgentDictEntry):
    """Holds participating coupling variables (consensus and exchange) of a single
    agent in ADMM. Used in the coordinator."""

    coup_vars: List[str] = dataclasses.field(default_factory=lambda: [])
    exchange_vars: List[str] = dataclasses.field(default_factory=lambda: [])


@dataclasses.dataclass
class ADMMParameters:
    """Collection of parameters which have to be shared across all agents in ADMM."""

    penalty_factor: float
    prediction_horizon: int
    time_step: float


@dataclasses.dataclass
class ADMMParticipation:
    """Helper class to organize ADMM participants."""

    source: al.Source
    ready: bool = False
    participating: bool = False


@dataclasses.dataclass
class CouplingVariable:
    """Holds information about a phy"""

    local_trajectories: Dict[al.Source, list] = dataclasses.field(default_factory=dict)
    mean_trajectory: list = dataclasses.field(default_factory=lambda: [0])
    delta_mean: np.ndarray = dataclasses.field(default_factory=lambda: np.array([0]))
    primal_residual: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0])
    )

    def _relevant_sources(self, sources: Iterable[al.Source]) -> set:
        if sources is None:
            sources = self.local_trajectories.keys()
        else:
            # the remaining sources are only agents that have this variable
            sources = set(self.local_trajectories.keys()).intersection(sources)
        return sources

    @property
    def participants(self):
        """Returns all agent sources that are registered to this coupling."""
        return list(self.local_trajectories.keys())

    def flat_locals(self, sources: Iterable[al.Source] = None) -> list[float]:
        """
        Returns the flattened array of all local variables and their multipliers.

        Args:
            sources: list of sources that should be included in the update.
                By default, all are included.

        Returns:
            flat lists of local variables and multipliers (locals, multipliers)
        """
        sources = self._relevant_sources(sources)
        if not sources:
            return []
        local_vars = list(chain([self.local_trajectories[ls] for ls in sources]))
        return local_vars

    def get_residual(self, rho: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the primal and dual residual of the last iteration as a tuple
        of flattened Arrays.
        Args:
            rho:

        Returns:
            (primal residual, dual residual)
        """
        primal_residual = self.primal_residual.flatten()
        dual_residual = (rho * self.delta_mean).flatten()
        return primal_residual, dual_residual


@dataclasses.dataclass
class ConsensusVariable(CouplingVariable):
    multipliers: Dict[al.Source, list] = dataclasses.field(default_factory=lambda: {})

    def update_mean_trajectory(self, sources: Iterable[al.Source] = None):
        """
        Calculates a new mean of this variable.

        Args:
            sources: List of sources that should be included in the update.
                If none is given, use all variables.
        """
        sources = self._relevant_sources(sources)
        if not sources:
            return
        lists = [self.local_trajectories[ls] for ls in sources]
        arr = np.array(lists)
        mean = np.mean(arr, axis=0)
        self.delta_mean = self.mean_trajectory - mean  # for residual
        self.mean_trajectory = list(mean)

    def update_multipliers(self, rho: float, sources: Iterable[al.Source] = None):
        """
        Performs the multiplier update.

        Args:
            rho: penalty parameter
            sources: list of sources that should be included in the update.
                By default, all are included.

        Returns:

        """
        sources = self._relevant_sources(sources)
        if not sources:
            return

        # create arrays for all trajectories and multipliers
        traj_list = [self.local_trajectories[ls] for ls in sources]
        mul_list = [self.multipliers[ls] for ls in sources]
        trajectories = np.array(traj_list)
        multipliers = np.array(mul_list)
        mean = np.array(self.mean_trajectory)

        # perform the update
        self.primal_residual = mean - trajectories
        new_multipliers = multipliers - rho * self.primal_residual

        # cast the updated multipliers back to their sources
        for i, src in enumerate(sources):
            self.multipliers[src] = new_multipliers[i, :].tolist()

    def flat_multipliers(self, sources: Iterable[al.Source] = None) -> list[float]:
        sources = self._relevant_sources(sources)
        if not sources:
            return []
        return list(chain([self.multipliers[ls] for ls in sources]))

    def shift_values_by_one(self, horizon: int):
        """Shifts the trajectories"""
        mean_traj = self.mean_trajectory
        shift_by = int(len(mean_traj) / horizon)
        self.mean_trajectory = mean_traj[shift_by:] + mean_traj[-shift_by:]
        mul_dict = self.multipliers
        for key, mul in mul_dict.items():
            mul_dict[key] = mul[shift_by:] + mul[-shift_by:]


@dataclasses.dataclass
class ExchangeVariable(CouplingVariable):
    diff_trajectories: Dict[al.Source, list[float]] = dataclasses.field(
        default_factory=dict
    )
    multiplier: list[float] = dataclasses.field(default_factory=list)

    def update_diff_trajectories(self, sources: Iterable[al.Source] = None):
        """
        Calculates a new mean of this variable.

        Args:
            sources: List of sources that should be included in the update.
                If none is given, use all variables.
        """
        sources = self._relevant_sources(sources)
        if not sources:
            return
        lists = [self.local_trajectories[ls] for ls in sources]
        arr = np.array(lists)
        mean = np.mean(arr, axis=0)
        self.delta_mean = self.mean_trajectory - mean  # for residual
        self.mean_trajectory = list(mean)
        for src in sources:
            self.diff_trajectories[src] = list(self.local_trajectories[src] - mean)

    def update_multiplier(self, rho: float):
        """
        Performs the multiplier update.

        Args:
            rho: penalty parameter

        Returns:

        """

        # perform the update
        self.primal_residual = np.array(self.mean_trajectory)
        self.multiplier = list(self.multiplier + rho * self.primal_residual)

    def shift_values_by_one(self, horizon: int):
        """Shifts the trajectories"""
        shift_by = int(len(self.multiplier) / horizon)
        self.multiplier = self.multiplier[shift_by:] + self.multiplier[-shift_by:]
        for key, diff in self.diff_trajectories.items():
            self.diff_trajectories[key] = diff[shift_by:] + diff[-shift_by:]


@dataclasses.dataclass
class StructuredValue:
    """Base Class to specify the structure of an AgentVariable Value. It will
    be efficiently sent and deserialized."""

    def to_json_bytes(self) -> bytes:
        """Serialize self to json bytes, can be used by the communicator."""
        return orjson.dumps(
            self, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS
        )

    @classmethod
    def from_json(cls, data: bytes):
        return cls(**orjson.loads(data))


@dataclasses.dataclass
class CoordinatorToAgent(StructuredValue):
    target: str
    mean_trajectory: Dict[str, list]
    multiplier: Dict[str, list]
    mean_diff_trajectory: Dict[str, list]
    exchange_multiplier: Dict[str, list]
    penalty_parameter: float


@dataclasses.dataclass
class AgentToCoordinator(StructuredValue):
    local_trajectory: Dict[str, np.ndarray]
    local_exchange_trajectory: Dict[str, np.ndarray]
