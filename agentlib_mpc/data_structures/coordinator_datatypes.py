import dataclasses
from dataclasses import dataclass, asdict
from aenum import Enum

import numpy as np

from agentlib.core import Source

# Parameter names
PREDICTION_HORIZON = "prediction_horizon"
TIME_STEP = "time_step"


# Communication names / aliases
REGISTRATION = "registration"
REGISTRATION_C2A = "registration_coordinator_to_agent"
REGISTRATION_A2C = "registration_agent_to_coordinator"
START_ITERATION = "startIteration"
START_ITERATION_C2A = "startIteration_coordinator_to_agent"
START_ITERATION_A2C = "startIteration_agent_to_coordinator"
OPTIMIZATION = "optimization"
OPTIMIZATION_C2A = "optimization_coordinator_to_agent"
OPTIMIZATION_A2C = "optimization_agent_to_coordinator"


class CoordinatorStatus(str, Enum):
    """Enum used to keep track of the status of a DMPC-Coordinator."""

    sleeping = "sleeping"
    init_iterations = "init_iterations"
    optimization = "optimization"
    updating = "updating"


class AgentStatus(str, Enum):
    """Enum used within a DMPC-Coordinator to keep track of the statuses of its
    participating agents."""

    pending = "pending"  # agent is not yet fully registered
    standby = "standby"  # agent is fully registered but not initialized for opt
    ready = "ready"  # agent is ready to start a local optimization
    busy = "busy"  # agent is performing task


@dataclass
class OptimizationData:
    """
    Contains specific variables (or time series) of the agent
    """

    x: np.array = dataclasses.field(default_factory=lambda: np.array([]))
    u: np.array = dataclasses.field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict:
        inst_dict = asdict(self)
        for key, val in inst_dict.items():
            if isinstance(val, np.ndarray):
                inst_dict[key] = np.array2string(val)
        return inst_dict

    @classmethod
    def from_dict(cls, data: dict):
        for key, val in data.items():
            try:
                data[key] = np.frombuffer(val)
            except (ValueError, TypeError):
                pass
        return cls(**data)


@dataclass
class RegistrationMessage:
    """Dataclass structuring the communication during registration between a
    participating agent and the coordinator in DMPC."""

    status: AgentStatus = None
    opts: dict = None
    source: Source = None
    coupling: list = None


# EXECUTION
@dataclass
class AgentDictEntry:
    """Dataclass holding the status of a participating agent in DMPC."""

    name: str
    optimization_data = OptimizationData()
    status: AgentStatus = AgentStatus.pending
