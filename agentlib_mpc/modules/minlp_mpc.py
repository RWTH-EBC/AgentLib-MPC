import logging

from pydantic import field_validator, Field, FieldValidationInfo
from rapidfuzz import process, fuzz

from agentlib.core import AgentVariable
from agentlib_mpc.data_structures import mpc_datamodels
from agentlib_mpc.data_structures.mpc_datamodels import MINLPVariableReference
from agentlib_mpc.modules.mpc import BaseMPCConfig, BaseMPC

logger = logging.getLogger(__name__)


class MINLPMPCConfig(BaseMPCConfig):
    """
    Pydantic data model for MPC configuration parser
    """

    # AgentVariables for the controls to be optimized
    binary_controls: mpc_datamodels.MPCVariables = Field(
        default=[], description="List of all binary control variables of the MPC. "
    )

    r_del_u: dict[str, float] = Field(
        default={},
        description="Weights that are applied to the change in control variables.",
    )

    @field_validator("binary_controls")
    @classmethod
    def validate_binary_bounds(cls, binary_controls: mpc_datamodels.MPCVariables):
        """Assures all binary variables have 0 and 1 as boundaries."""
        for bc in binary_controls:
            if bc.ub == 1 and bc.lb == 0:
                continue
            logger.warning(
                f"Binary variable {bc.name} does not have bounds '0, 1'. This will be"
                f" automatically changed."
            )
            bc.ub = 1
            bc.lb = 0
        return binary_controls

    @field_validator("r_del_u")
    def check_r_del_u_in_controls(
            cls, r_del_u: dict[str, float], info: FieldValidationInfo
    ):
        """Ensures r_del_u is only set for control variables."""
        controls = {ctrl.name for ctrl in
                    info.data["controls"] + info.data["binary_controls"]}
        for name in r_del_u:
            if name in controls:
                # everything is fine
                continue

            # raise error
            matches = process.extract(query=name, choices=controls, scorer=fuzz.WRatio)
            matches = [m[0] for m in matches]
            raise ValueError(
                f"Tried to specify control change weight for {name}. However, "
                f"{name} is not in the set of control variables. Did you mean one "
                f"of these? {', '.join(matches)}"
            )
        return r_del_u


class MINLPMPC(BaseMPC):
    config: MINLPMPCConfig

    def _after_config_update(self):
        self._internal_variables = self._create_internal_variables()
        super()._after_config_update()

    def _setup_var_ref(self) -> mpc_datamodels.VariableReferenceT:
        return MINLPVariableReference.from_config(self.config)

    def collect_variables_for_optimization(
        self, var_ref: mpc_datamodels.VariableReference = None
    ) -> dict[str, AgentVariable]:
        """Gets all variables noted in the var ref and puts them in a flat
        dictionary."""
        if var_ref is None:
            var_ref = self.var_ref

        # config variables
        variables = {v: self.get(v) for v in var_ref.all_variables()}

        return {**variables, **self._internal_variables}

    def assert_mpc_variables_are_in_model(self):
        """
        Checks whether all variables of var_ref are contained in the model.
        Returns names of model variables not contained in the var_ref,
        sorted by keys: 'states', 'inputs', 'outputs', 'parameters'.
        """

        # arguments for validation function:
        # (key in var_ref, model names, str for head error message)
        args = [
            (
                "states",
                self.model.get_state_names(),
                "Differential variables / States",
            ),
            ("controls", self.model.get_input_names(), "Controls"),
            ("binary_controls", self.model.get_input_names(), "Binary Controls"),
            ("inputs", self.model.get_input_names(), "Inputs"),
            ("outputs", self.model.get_output_names(), "Outputs"),
            ("parameters", self.model.get_parameter_names(), "Parameters"),
        ]

        # perform validations and make a dictionary of unassigned variables
        unassigned_by_mpc_var = {
            key: self.assert_subset(self.var_ref.__dict__[key], names, message)
            for key, names, message in args
        }

        # fix unassigned values for inputs
        intersection_input = set.intersection(
            unassigned_by_mpc_var["controls"],
            unassigned_by_mpc_var["inputs"],
            unassigned_by_mpc_var["binary_controls"],
        )

        # return dict should have model variables as keys, not mpc variables
        unassigned_by_model_var = {
            "states": unassigned_by_mpc_var["states"],
            "inputs": intersection_input,
            "outputs": unassigned_by_mpc_var["outputs"],
            "parameters": unassigned_by_mpc_var["parameters"],
        }

        return unassigned_by_model_var

    def set_actuation(self, solution):
        """Takes the solution from optimization backend and sends the first
        step to AgentVariables."""
        super().set_actuation(solution)
        for b_control in self.var_ref.binary_controls:
            # take the first entry of the control trajectory
            actuation = solution[b_control][0]
            self.set(b_control, actuation)

    def _create_internal_variables(self) -> dict[str, AgentVariable]:
        """Creates a reference of all internal variables that are used for the MPC,
        but not shared as AgentVariables.

        Currently, this includes:
           - Weights for control change (r_del_u)
        """
        r_del_u: dict[str, mpc_datamodels.MPCVariable] = {}
        for control in self.config.controls:
            r_del_u_name = mpc_datamodels.r_del_u_convention(control.name)
            var = mpc_datamodels.MPCVariable(name=r_del_u_name)
            r_del_u[r_del_u_name] = var
            if control.name in self.config.r_del_u:
                var.value = self.config.r_del_u[control.name]
            else:
                var.value = 0

        return r_del_u
