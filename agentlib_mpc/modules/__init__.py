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
    "data_source": ModuleImport(
        module_path="agentlib_mpc.modules.data_source", class_name="DataSource"
    ),
    "mpc_basic": ModuleImport(
        module_path="agentlib_mpc.modules.mpc", class_name="BaseMPC"
    ),
    "mpc": ModuleImport(module_path="agentlib_mpc.modules.mpc_full", class_name="MPC"),
    "minlp_mpc": ModuleImport(
        module_path="agentlib_mpc.modules.minlp_mpc", class_name="MINLPMPC"
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
        module_path="agentlib_mpc.modules.ml_model_training.ann_trainer",
        class_name="ANNTrainer",
    ),
    "gpr_trainer": ModuleImport(
        module_path="agentlib_mpc.modules.ml_model_training.gpr_trainer",
        class_name="GPRTrainer",
    ),
    "linreg_trainer": ModuleImport(
        module_path="agentlib_mpc.modules.ml_model_training.linreg_trainer",
        class_name="LinRegTrainer",
    ),
    "ml_model_simulator": ModuleImport(
        module_path="agentlib_mpc.modules.ml_model_simulator",
        class_name="MLModelSimulator",
    ),
    "set_point_generator": ModuleImport(
        module_path="agentlib_mpc.modules.ml_model_training.setpoint_generator",
        class_name="SetPointGenerator",
    ),
    "mhe": ModuleImport(
        module_path="agentlib_mpc.modules.estimation.mhe", class_name="MHE"
    ),
}
