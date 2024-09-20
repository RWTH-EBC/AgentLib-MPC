"""
Package containing models for agentlib_mpc.
"""
from agentlib.utils.plugin_import import ModuleImport

MODEL_TYPES = {
    "casadi": ModuleImport(
        import_path="agentlib_mpc.models.casadi_model", class_name="CasadiModel"
    ),
    "casadi_ml": ModuleImport(
        import_path="agentlib_mpc.models.casadi_ml_model", class_name="CasadiMLModel"
    ),
    "grampc": ModuleImport(
        import_path="agentlib_mpc.models.grampc_model", class_name="GrampcModel"
    ),
}
