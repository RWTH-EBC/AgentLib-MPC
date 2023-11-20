from agentlib_mpc.modules.mpc_full import MPC, MPCConfig


class DistributedMPCConfig(MPCConfig):
    """
    Base config class with common configurations
    """


class DistributedMPC(MPC):
    """Base class which defines common interfaces among all
    distributed mpc approaches (either optimization based,
    game theory based or some other)."""

    config: DistributedMPCConfig
