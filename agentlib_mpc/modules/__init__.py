"""
This package contains all modules for the
distributed model predictive control using multi agent systems.

It contains classes for local optimization and global coordination.
"""
import importlib


class ModuleImport:
    def __init__(self, module_path: str, class_name: str):
        self.module_path = module_path
        self.class_name = class_name

    def import_class(self):
        module = importlib.import_module(self.module_path)
        return getattr(module, self.class_name)


MODULE_TYPES = {
    "mpc_basic": ModuleImport(
        module_path="agentlib_mpc.modules.mpc", class_name="BaseMPC"
    ),
    "mpc": ModuleImport(module_path="agentlib_mpc.modules.mpc_full", class_name="MPC"),
    "miqp_mpc": ModuleImport(
        module_path="agentlib_mpc.modules.miqp_mpc", class_name="MIQPMPC"
    ),
    "admm": ModuleImport(
        module_path="agentlib_mpc.modules.dmpc.admm.admm", class_name="ADMM"
    ),
    "admm_local": ModuleImport(
        module_path="agentlib_mpc.modules.dmpc.admm.admm", class_name="LocalADMM"
    ),
    "admm_coordinated": ModuleImport(
        module_path="agentlib_mpc.modules.dmpc.admm.admm_coordinated",
        class_name="CoordinatedADMM",
    ),
    "admm_coordinator": ModuleImport(
        module_path="agentlib_mpc.modules.dmpc.admm.admm_coordinator",
        class_name="ADMMCoordinator",
    ),
    "ann_trainer": ModuleImport(
        module_path="agentlib_mpc.modules.ml_model_training.ml_model_trainer",
        class_name="ANNTrainer",
    ),
    "gpr_trainer": ModuleImport(
        module_path="agentlib_mpc.modules.ml_model_training.ml_model_trainer",
        class_name="GPRTrainer",
    ),
    "linreg_trainer": ModuleImport(
        module_path="agentlib_mpc.modules.ml_model_training.ml_model_trainer",
        class_name="LinRegTrainer",
    ),
    "ann_simulator": ModuleImport(
        module_path="agentlib_mpc.modules.ann_simulator",
        class_name="MLModelSimulator",
    ),
    "set_point_generator": ModuleImport(
        module_path="agentlib_mpc.modules.ml_model_training.setpoint_generator",
        class_name="SetPointGenerator",
    ),
}


# MODULE_TYPES = {
#     "mpc": ModuleImport(file="mpc/mpc.py", class_name="MPC"),
#     "admm": ModuleImport(file="mpc/dmpc/admm/admm.py", class_name="ADMM"),
#     "admm_local": ModuleImport(file="mpc/dmpc/admm/admm.py", class_name="LocalADMM"),
#     "admm_coordinated": ModuleImport(
#         file="mpc/dmpc/admm/admm_coordinated.py", class_name="CoordinatedADMM"
#     ),
#     "admm_coordinator": ModuleImport(
#         file="mpc/dmpc/admm/admm_coordinator.py", class_name="ADMMCoordinator"
#     ),
# }
