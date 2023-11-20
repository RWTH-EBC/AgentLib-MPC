from __future__ import annotations

import dataclasses

import casadi as ca

from agentlib_mpc.models.casadi_model import CasadiVariable


@dataclasses.dataclass(frozen=True)
class OptimizationQuantity:
    name: str
    full_symbolic: ca.MX  # used in complex cost functions for admm etc
    dim: int
    ref_names: tuple[str]  # used in get_mpc_inputs
    full_names: tuple[str]  # used in create_res_format
    use_in_stage_function: bool

    def __hash__(self):
        return hash(self.name)


def _check_ref_in_full(ref: list[str], full_names: list[str]):
    diff = set(ref).difference(full_names)
    if diff:
        raise ValueError(
            f"The variables from the variable ref are not a subset of the model "
            f"variables. The following variables are wrong: {diff}"
        )


@dataclasses.dataclass(frozen=True)
class OptimizationVariable(OptimizationQuantity):
    input_map: ca.Function  # get mpc inputs
    output_map: ca.Function  # get mpc outputs
    use_in_stage_function: bool
    binary: bool

    @classmethod
    def declare(
        cls,
        denotation: str,
        variables: list[CasadiVariable],
        ref_list: list[str],
        use_in_stage_function: bool = True,
        assert_complete: bool = False,
        binary: bool = False,
    ) -> OptimizationVariable:
        """
        Declares a group of optimization variables that serve a purpose in
        the optimization problem. Typical groups are states, the control
        inputs or slack variables.

        Args:
            binary: Flag, whether these variables are binary
            denotation: The key of the variable, e.g. 'X', 'U', etc. Use
                this key in the discretization function to add the variable at
                different stages of the optimization problem. The optimal value
                of these variables will also be mapped to this key.
            variables: A list of
                CasadiVariables or an MX/SX vector including all variables
                within this category.
            ref_list: A list of names indicating which variables
                in full_list are AgentVariables and need to be updated before
                each optimization.
            use_in_stage_function: If False, the variable is not
                added to the stage function. If True, the variable needs to be
                provided to the stage function at every point in the
                discretization function.
            assert_complete: If True, throws an error if the ref_list does
                not contain all variables.
        """
        full_symbolic = []
        full_names = []
        ref_symbolic = []
        lb_full = []
        lb_ref = []
        ub_full = []
        ub_ref = []
        ref_list_ordered = []

        for var in variables:
            name = var.name
            if assert_complete and name not in ref_list:
                raise ValueError(
                    f"The variable {name} which is defined in the model "
                    f" has to be defined in the ModuleConfig!"
                )

            full_symbolic.append(var.sym)
            full_names.append(name)

            if name in ref_list:
                lb = ca.MX.sym(f"lb_{denotation}")
                ub = ca.MX.sym(f"lb_{denotation}")
                lb_ref.append(lb)
                ub_ref.append(ub)
                ref_symbolic.append(var.sym)
                ref_list_ordered.append(name)
                lb_full.append(lb)
                ub_full.append(ub)
            else:
                lb_full.append(var.lb)
                ub_full.append(var.ub)

        full_symbolic = ca.vertcat(*full_symbolic)

        # create functions that map between model variable vectors (so all variables)
        # and variables from the var_ref (only the ones specified in the user config)
        input_mapping = ca.Function(
            f"par_map_{denotation}",
            [ca.vertcat(*lb_ref), ca.vertcat(*ub_ref)],
            [ca.vertcat(*lb_full), ca.vertcat(*ub_full)],
            ["lb_ref", "ub_ref"],
            [f"lb_{denotation}", f"ub_{denotation}"],
        )
        output_mapping = ca.Function(
            f"par_map_{denotation}",
            [full_symbolic],
            [ca.vertcat(*ref_symbolic)],
            [denotation],
            ["ref"],
        )

        dimension = full_symbolic.shape[0]
        _check_ref_in_full(ref_list, full_names)
        return cls(
            name=denotation,
            full_symbolic=full_symbolic,
            dim=dimension,
            ref_names=tuple(ref_list_ordered),
            full_names=tuple(full_names),
            use_in_stage_function=use_in_stage_function,
            input_map=input_mapping,
            output_map=output_mapping,
            binary=binary,
        )

    def __hash__(self):
        return hash(self.name)


@dataclasses.dataclass(frozen=True)
class OptimizationParameter(OptimizationQuantity):
    full_with_defaults: ca.MX
    add_default_values: ca.Function

    @classmethod
    def declare(
        cls,
        denotation: str,
        variables: list[CasadiVariable],
        ref_list: list[str],
        use_in_stage_function=True,
        assert_complete: bool = False,
    ):
        """
        Declares a group of optimization parameters that serve a purpose in
        the optimization problem. Typical groups are uncontrollable inputs or
        physical parameters.

        Args:
            denotation: The key of the variable, e.g. 'p', 'd', etc. Use this
                key in the discretization function to add the parameter at
                different stages of the optimization problem.
            variables: A list of CasadiVariables including all parameters
                within this category.
            ref_list: A list of names indicating which parameters in full_list
                are AgentVariables and need to be updated before each
                optimization.
            use_in_stage_function: If False, the parameter is not added to the
                stage function. If True, the variable needs to be provided to
                the stage function at every point in the discretization function.
            assert_complete: If True, throws an error if the ref_list does
                not contain all variables.
        """
        provided = []
        full_with_defaults = []
        full_symbolic = []
        full_names = []
        ref_list_ordered = []
        for var in variables:
            name = var.name
            if assert_complete:
                assert name in ref_list, (
                    f"The variable {name} which is defined in the model "
                    f" has to be defined in the ModuleConfig!"
                )

            full_symbolic.append(var.sym)
            full_names.append(name)
            if name in ref_list:
                full_with_defaults.append(var.sym)
                provided.append(var.sym)
                ref_list_ordered.append(name)
            else:
                if var.value is None:
                    raise ValueError(
                        f"Parameter '{name}' is not declared in the module "
                        f"config. Tried using default from model "
                        f" but it was 'None'."
                    )
                full_with_defaults.append(var.value)
        full_with_defaults = ca.vertcat(*full_with_defaults)

        add_default_values = ca.Function(
            f"par_map_{denotation}",
            [ca.vertcat(*provided)],
            [full_with_defaults],
            ["ref"],
            [denotation],
        )
        _check_ref_in_full(ref_list, full_names)
        return OptimizationParameter(
            name=denotation,
            full_with_defaults=full_with_defaults,
            full_symbolic=ca.vertcat(*full_symbolic),
            dim=full_with_defaults.shape[0],
            ref_names=tuple(ref_list_ordered),
            full_names=tuple(full_names),
            use_in_stage_function=use_in_stage_function,
            add_default_values=add_default_values,
        )

    def __hash__(self):
        return hash(self.name)
