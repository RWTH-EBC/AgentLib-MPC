import importlib

from pydantic import BaseModel


class BackendImport(BaseModel):
    """
    Data-Class to import a given python file
    from ``import_path`` and load the given
    ``class_name``
    """

    import_path: str
    class_name: str

    def __call__(self, *args, **kwargs):
        """Import the Module with class_name from the import path"""
        module = importlib.import_module(self.import_path)
        cls = getattr(module, self.class_name)
        return cls(*args, **kwargs)


backend_types = {
    "casadi_basic": BackendImport(
        import_path="agentlib_mpc.optimization_backends.casadi_.basic",
        class_name="CasADiBaseBackend",
    ),
    "casadi": BackendImport(
        import_path="agentlib_mpc.optimization_backends.casadi_.full",
        class_name="CasADiFullBackend",
    ),
    "casadi_admm": BackendImport(
        import_path="agentlib_mpc.optimization_backends.casadi_.admm",
        class_name="CasADiADMMBackend",
    ),
    "casadi_minlp": BackendImport(
        import_path="agentlib_mpc.optimization_backends.casadi_.minlp",
        class_name="CasADiMINLPBackend",
    ),
    "casadi_cia": BackendImport(
        import_path="agentlib_mpc.optimization_backends.casadi_.minlp_cia",
        class_name="CasADiCIABackend",
    ),
    "casadi_ml": BackendImport(
        import_path="agentlib_mpc.optimization_backends.casadi_.casadi_ml",
        class_name="CasADiBBBackend",
    ),
    "casadi_admm_ml": BackendImport(
        import_path="agentlib_mpc.optimization_backends.casadi_.casadi_admm_ml",
        class_name="CasADiADMMBackend_ML",
    ),
    "grampc": BackendImport(
        import_path="agentlib_mpc.optimization_backends.grampc_.grampc_backend",
        class_name="GrampcBackend",
    ),
}


uninstalled_backend_types = {}

try:
    pass
except ImportError:
    uninstalled_backend_types.update()
