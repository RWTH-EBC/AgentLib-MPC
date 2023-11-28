"""
Package containing models for agentlib_mpc.
"""
from agentlib.utils.plugin_import import ModuleImport

MODEL_TYPES = {
    "casadi": ModuleImport(
        import_path="agentlib_mpc.models.casadi_model", class_name="CasadiModel"
    ),
}
