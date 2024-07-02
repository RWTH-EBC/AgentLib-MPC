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

        # Parameters that are constant over the horizon
        const_par = self.add_opt_par(sys.model_parameters)

        # Formulate the NLP
        # loop over prediction horizon
        while self.k < n:
            # New NLP variable for the control
            uk = self.add_opt_var(sys.controls)
            wk = self.add_opt_var(sys.binary_controls)

            # New parameter for inputs
            dk = self.add_opt_par(sys.non_controlled_inputs)

            # perform inner collocation loop
            opt_vars_inside_inner = [sys.algebraics, sys.outputs]
            opt_pars_inside_inner = []

            constant_over_inner = {
                sys.controls: uk,
                sys.non_controlled_inputs: dk,
                sys.model_parameters: const_par,
                sys.binary_controls: wk,
            }
            xk_end = self._collocation_inner_loop(
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
            self.add_constraint(xk_end - xk)


class CasADiMINLPBackend(CasADiBackend):
    """
    Class doing optimization of ADMM subproblems with CasADi.
    """

    system_type = CasadiMINLPSystem
    discretization_types = {DiscretizationMethod.collocation: DirectCollocation}
    system: CasadiMINLPSystem
