import casadi as ca
import numpy as np

from agentlib_mpc.data_structures.casadi_utils import DiscretizationMethod
from agentlib_mpc.data_structures.mpc_datamodels import MINLPVariableReference
from agentlib_mpc.models.casadi_model import CasadiModel
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationVariable,
)

from agentlib_mpc.optimization_backends.casadi_ import basic
from agentlib_mpc.optimization_backends.casadi_.core.casadi_backend import CasADiBackend

# todo: All the names are minlp, but this is actually minlp capable


class CasadiMINLPSystem(basic.BaseSystem):
    binary_controls: OptimizationVariable

    def __init__(self):
        super().__init__()
        self.is_linear = False

    def initialize(self, model: CasadiModel, var_ref: MINLPVariableReference):
        self.binary_controls = OptimizationVariable.declare(
            denotation="w",
            variables=model.get_inputs(var_ref.binary_controls),
            ref_list=var_ref.binary_controls,
            assert_complete=True,
            binary=True,
        )
        super().initialize(model=model, var_ref=var_ref)
        self.is_linear = self._is_minlp()

    def _is_minlp(self) -> bool:
        inputs = ca.vertcat(*(v.full_symbolic for v in self.variables))
        parameters = ca.vertcat(
            *(v.full_symbolic for v in self.parameters if v.use_in_stage_function)
        )
        test_params = [1] * ca.vertcat(
            *(
                v.add_default_values()[v.name]
                for v in self.parameters
                if v.use_in_stage_function
            )
        )
        ode = self.ode
        constraints = self.model_constraints.function
        outputs = ca.vertcat(ode, constraints)
        jac = ca.jacobian(outputs, inputs)
        print(jac)
        test_input = [0] * inputs.shape[0]
        jac_func = ca.Function(
            "jac_func",
            [inputs, parameters],
            [jac],
            ["inputs", "parameters"],
            ["jacobian"],
        )
        test2 = np.array(test_input) + 0.5
        return jac_func(test_input, test_params) == jac_func(test2, test_params)


class DirectCollocation(basic.DirectCollocation):
    def _discretize(self, sys: CasadiMINLPSystem):
        """
        Defines a direct collocation discretization.
        # pylint: disable=invalid-name
        """

        # setup the polynomial base
        collocation_matrices = self._collocation_polynomial()

        # shorthands
        n = self.options.prediction_horizon
        ts = self.options.time_step

        # Initial State
        x0 = self.add_opt_par(sys.initial_state)
        xk = self.add_opt_var(sys.states, lb=x0, ub=x0, guess=x0)

        # Initialize control tracking for delta_u objectives
        uk = None
        if hasattr(sys, 'last_control'):
            uk = self.add_opt_par(sys.last_control)

        # Parameters that are constant over the horizon
        const_par = self.add_opt_par(sys.model_parameters)

        # Handle delta_u objectives - only for new objective system
        delta_u_objectives = []
        if (hasattr(sys, 'model') and
                hasattr(sys.model, 'objective') and
                sys.model.objective is not None):
            try:
                delta_u_objectives = sys.model.objective.get_delta_u_objectives()
            except (AttributeError, Exception) as e:
                self.logger.warning(f"Failed to get delta_u_objectives: {str(e)}")

        control_map = {}
        for i, control_name in enumerate(sys.controls.ref_names):
            control_map[control_name] = i

        # Formulate the NLP
        # loop over prediction horizon
        while self.k < n:
            # New NLP variable for the control
            u_prev = uk
            uk = self.add_opt_var(sys.controls)
            wk = self.add_opt_var(sys.binary_controls)

            # penalty for control change between time steps (only for new objective system)
            if delta_u_objectives and u_prev is not None:
                for delta_obj in delta_u_objectives:
                    control_name = delta_obj.get_control_name()
                    if control_name in control_map:
                        idx = control_map[control_name]
                        control_prev = u_prev[idx]
                        control_curr = uk[idx]
                        delta = control_curr - control_prev

                        if hasattr(delta_obj.weight, 'sym'):
                            param_found = False
                            for i, param_name in enumerate(sys.model_parameters.ref_names):
                                if param_name == delta_obj.weight.name:
                                    weight_value = const_par[i]
                                    param_found = True
                                    break

                            if not param_found:
                                raise ValueError(f"Parameter {delta_obj.weight.name} not found in model parameters")
                        else:
                            weight_value = delta_obj.weight

                        self.objective_function += weight_value ** 2 * delta ** 2

            # perform inner collocation loop
            opt_vars_inside_inner = [sys.algebraics, sys.outputs]
            opt_pars_inside_inner = [sys.non_controlled_inputs]

            constant_over_inner = {
                sys.controls: uk,
                sys.model_parameters: const_par,
                sys.binary_controls: wk,
            }
            xk_end, constraints = self._collocation_inner_loop(
                collocation=collocation_matrices,
                state_at_beginning=xk,
                states=sys.states,
                opt_vars=opt_vars_inside_inner,
                opt_pars=opt_pars_inside_inner,
                const=constant_over_inner,
            )

            # increment loop counter and time
            self.k += 1
            self.pred_time = ts * self.k

            # New NLP variable for differential state at end of interval
            xk = self.add_opt_var(sys.states)

            # Add continuity constraint
            self.add_constraint(xk - xk_end, gap_closing=True)

            # add collocation constraints later for fatrop
            for constraint in constraints:
                self.add_constraint(*constraint)


class CasADiMINLPBackend(CasADiBackend):
    """
    Class doing optimization of ADMM subproblems with CasADi.
    """

    system_type = CasadiMINLPSystem
    discretization_types = {DiscretizationMethod.collocation: DirectCollocation}
    system: CasadiMINLPSystem
