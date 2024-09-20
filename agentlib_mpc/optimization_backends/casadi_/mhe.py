import dataclasses
import logging
from typing import Union

import casadi as ca
import numpy as np
import pandas as pd
from scipy import interpolate

from agentlib_mpc.data_structures.casadi_utils import (
    Constraint,
    LB_PREFIX,
    UB_PREFIX,
    DiscretizationMethod,
    SolverFactory,
)
from agentlib_mpc.data_structures.mpc_datamodels import MHEVariableReference
from agentlib_mpc.models.casadi_model import CasadiModel, CasadiInput
from agentlib_mpc.optimization_backends.casadi_.core.casadi_backend import CasADiBackend
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationQuantity,
    OptimizationVariable,
    OptimizationParameter,
)
from agentlib_mpc.optimization_backends.casadi_.core.discretization import (
    Discretization,
)
from agentlib_mpc.optimization_backends.casadi_.core.system import System


logger = logging.getLogger(__name__)


class MHESystem(System):
    # variables
    estimated_states: OptimizationVariable
    estimated_inputs: OptimizationVariable
    estimated_parameters: OptimizationVariable
    algebraics: OptimizationVariable
    outputs: OptimizationVariable

    # parameters
    measured_states: OptimizationParameter
    known_inputs: OptimizationParameter
    known_parameters: OptimizationParameter

    # dynamics
    model_constraints: Constraint
    cost_function: ca.MX
    ode: ca.MX

    def initialize(self, model: CasadiModel, var_ref: MHEVariableReference):
        # define variables
        self.states = OptimizationVariable.declare(
            denotation="states",
            variables=model.get_states(var_ref.states),
            ref_list=var_ref.states,
            assert_complete=True,
        )
        self.estimated_inputs = OptimizationVariable.declare(
            denotation="estimated_inputs",
            variables=model.get_inputs(var_ref.estimated_inputs),
            ref_list=var_ref.estimated_inputs,
            assert_complete=True,
        )
        self.estimated_parameters = OptimizationVariable.declare(
            denotation="estimated_parameters",
            variables=model.get_parameters(var_ref.estimated_parameters),
            ref_list=var_ref.estimated_parameters,
        )
        self.algebraics = OptimizationVariable.declare(
            denotation="algebraics",
            variables=model.algebraics + model.auxiliaries,
            ref_list=[],
        )
        self.outputs = OptimizationVariable.declare(
            denotation="outputs",
            variables=model.outputs,
            ref_list=var_ref.outputs,
        )

        # define parameters
        # self.measured_states = OptimizationParameter.declare(
        #     denotation="measured_states",
        #     variables=model.get_states(var_ref.measured_states),
        #     ref_list=var_ref.measured_states,
        #     assert_complete=True,
        # )
        self.known_inputs = OptimizationParameter.declare(
            denotation="known_inputs",
            variables=model.get_inputs(var_ref.known_inputs),
            ref_list=var_ref.known_inputs,
            assert_complete=True,
        )
        known_parameter_names = set(model.get_parameter_names()) - set(
            var_ref.estimated_parameters
        )
        self.known_parameters = OptimizationParameter.declare(
            denotation="known_parameters",
            variables=model.get_parameters(list(known_parameter_names)),
            ref_list=var_ref.known_parameters,
            assert_complete=False,
        )

        measured_state_names = ["measured_" + state for state in var_ref.states]
        self.measured_states = OptimizationParameter.declare(
            denotation="measured_states",
            variables=[CasadiInput(name=name) for name in measured_state_names],
            ref_list=measured_state_names,
        )
        weights_state_names = ["weight_" + state for state in var_ref.states]
        self.weights_states = OptimizationParameter.declare(
            denotation="weight_states",
            variables=[CasadiInput(name=name) for name in weights_state_names],
            ref_list=weights_state_names,
        )

        # add admm terms to objective function
        objective: ca.MX = 0
        for i in range(len(var_ref.states)):
            states = self.states.full_symbolic[i]
            measured_states = self.measured_states.full_symbolic[i]
            weights = self.weights_states.full_symbolic[i]
            objective += weights * (states - measured_states) ** 2

        # dynamics
        self.ode = model.system
        self.cost_function = objective
        self.model_constraints = Constraint(
            function=ca.vertcat(*[c.function for c in model.constraints]),
            lb=ca.vertcat(*[c.lb for c in model.constraints]),
            ub=ca.vertcat(*[c.ub for c in model.constraints]),
        )


@dataclasses.dataclass
class CollocationMatrices:
    order: int
    root: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


class DirectCollocation(Discretization):
    def _discretize(self, sys: MHESystem):
        """
        Defines a direct collocation discretization.
        # pylint: disable=invalid-name
        """

        # setup the polynomial base
        collocation_matrices = self._collocation_polynomial()

        # shorthands
        n = self.options.prediction_horizon
        ts = self.options.time_step
        start_time = -n * ts
        self.pred_time = start_time

        # Initial State
        x_est_k = self.add_opt_var(sys.states)

        # Parameters that are constant over the horizon
        known_pars = self.add_opt_par(sys.known_parameters)
        estimated_pars = self.add_opt_var(sys.estimated_parameters)
        weights = self.add_opt_par(sys.weights_states)

        # Formulate the NLP
        # loop over prediction horizon
        while self.k < n:
            # New NLP variable for the control
            inp_known = self.add_opt_par(sys.known_inputs)
            inp_est = self.add_opt_var(sys.estimated_inputs)

            # perform inner collocation loop
            opt_vars_inside_inner = [sys.outputs, sys.algebraics]
            opt_pars_inside_inner = [sys.measured_states]

            constant_over_inner = {
                sys.known_inputs: inp_known,
                sys.estimated_inputs: inp_est,
                sys.estimated_parameters: estimated_pars,
                sys.known_parameters: known_pars,
                sys.weights_states: weights,
            }
            xk_end = self._collocation_inner_loop(
                collocation=collocation_matrices,
                state_at_beginning=x_est_k,
                states=sys.states,
                opt_vars=opt_vars_inside_inner,
                opt_pars=opt_pars_inside_inner,
                const=constant_over_inner,
            )

            # increment loop counter and time
            self.k += 1
            self.pred_time = start_time + ts * self.k

            # New NLP variable for differential state at end of interval
            xk = self.add_opt_var(sys.states)

            # Add continuity constraint
            self.add_constraint(xk_end - xk)

    def _construct_stage_function(self, system: MHESystem):
        """
        Combine information from the model and the var_ref to create CasADi
        functions which describe the system dynamics and constraints at each
        stage of the optimization problem. Sets the stage function. It has
        all mpc variables as inputs, sorted by denotation (declared in
        self.declare_quantities) and outputs ode, cost function and 3 outputs
        per constraint (constraint, lb_constraint, ub_constraint).

        In the basic case, it has the form:
        CasadiFunction: ['x', 'z', 'u', 'y', 'd', 'p'] ->
            ['ode', 'cost_function', 'model_constraints',
            'ub_model_constraints', 'lb_model_constraints']

        Args:
            system
        """
        all_system_quantities: dict[str, OptimizationQuantity] = {
            var.name: var for var in system.quantities
        }
        constraints = {"model_constraints": system.model_constraints}

        inputs = [
            q.full_symbolic
            for q in all_system_quantities.values()
            if q.use_in_stage_function
        ]
        input_denotations = [
            q.name
            for denotation, q in all_system_quantities.items()
            if q.use_in_stage_function
        ]

        # aggregate constraints
        constraints_func = [c.function for c in constraints.values()]
        constraints_lb = [c.lb for c in constraints.values()]
        constraints_ub = [c.ub for c in constraints.values()]
        constraint_denotations = list(constraints.keys())
        constraint_lb_denotations = [LB_PREFIX + k for k in constraints]
        constraint_ub_denotations = [UB_PREFIX + k for k in constraints]

        # aggregate outputs
        outputs = [
            system.ode,
            system.cost_function,
            *constraints_func,
            *constraints_lb,
            *constraints_ub,
        ]
        output_denotations = [
            "ode",
            "cost_function",
            *constraint_denotations,
            *constraint_lb_denotations,
            *constraint_ub_denotations,
        ]

        # function describing system dynamics and cost function
        self._stage_function = ca.Function(
            "f",
            inputs,
            outputs,
            # input handles to make kwarg use possible and to debug
            input_denotations,
            # output handles to make kwarg use possible and to debug
            output_denotations,
        )

    def initialize(self, system: MHESystem, solver_factory: SolverFactory):
        """Initializes the trajectory optimization problem, creating all symbolic
        variables of the OCP, the mapping function and the numerical solver."""
        self._construct_stage_function(system)
        super().initialize(system=system, solver_factory=solver_factory)

    def _collocation_inner_loop(
        self,
        state_at_beginning: ca.MX,
        collocation: CollocationMatrices,
        states: OptimizationVariable,
        opt_vars: list[OptimizationVariable],
        opt_pars: list[OptimizationParameter],
        const: dict[OptimizationQuantity, ca.MX],
    ) -> ca.MX:
        """
        Constructs the inner loop of a collocation discretization. Takes the

        Args
            collocation: The collocation matrices
            state_at_beginning: The casadi MX instance representing the state at the
                beginning of the collocation interval
            states: The OptimizationVariable representing the states
            opt_vars: The OptimizationVariables which should be defined at each
                collocation point
            par_vars: The OptimizationParameters which should be defined at each
                collocation point
            const: Variables or parameters to feed into the system function that are
                constant over the inner loop. Value is the current MX to be used.

        Returns:
            state_k_end[MX]: state at the end of collocation interval
        """
        constants = {var.name: mx for var, mx in const.items()}

        # remember time at start of collocation loop
        start_time = self.pred_time

        # shorthands
        ts = self.options.time_step

        # State variables at collocation points
        state_collocation = []
        opt_vars_collocation = []
        opt_pars_collocation = []

        # add variables at collocation points
        for j in range(collocation.order):  # d is collocation order
            # set time
            self.pred_time = start_time + collocation.root[j + 1] * ts

            # differential state
            state_kj = self.add_opt_var(states, post_den=f"_{j}")
            state_collocation.append(state_kj)

            opt_vars_collocation.append({})
            for opt_var in opt_vars:
                var_kj = self.add_opt_var(opt_var, post_den=f"_{j}")
                opt_vars_collocation[-1].update({opt_var.name: var_kj})

            opt_pars_collocation.append({})
            for opt_par in opt_pars:
                par_kj = self.add_opt_par(opt_par, post_den=f"_{j}")
                opt_pars_collocation[-1].update({opt_par.name: par_kj})

        # Loop over collocation points
        state_k_end = collocation.D[0] * state_at_beginning
        for j in range(1, collocation.order + 1):
            # Expression for the state derivative at the collocation point
            xp = collocation.C[0, j] * state_at_beginning
            for r in range(collocation.order):
                xp = xp + collocation.C[r + 1, j] * state_collocation[r]

            stage = self._stage_function(
                **{states.name: state_collocation[j - 1]},
                **opt_pars_collocation[j - 1],
                **opt_vars_collocation[j - 1],
                **constants,
            )

            self.add_constraint(ts * stage["ode"] - xp)

            # Append inequality constraints
            self.add_constraint(
                stage["model_constraints"],
                lb=stage["lb_model_constraints"],
                ub=stage["ub_model_constraints"],
            )

            # Add contribution to the end state
            state_k_end = state_k_end + collocation.D[j] * state_collocation[j - 1]

            # Add contribution to quadrature function
            self.objective_function += collocation.B[j] * stage["cost_function"] * ts

        return state_k_end

    def _collocation_polynomial(self) -> CollocationMatrices:
        """Returns the matrices needed for direct collocation discretization."""
        # Degree of interpolating polynomial
        d = self.options.collocation_order
        polynomial = self.options.collocation_method

        # Get collocation points
        tau_root = np.append(0, ca.collocation_points(d, polynomial))

        # Coefficients of the collocation equation
        C = np.zeros((d + 1, d + 1))

        # Coefficients of the continuity equation
        D = np.zeros(d + 1)

        # Coefficients of the quadrature function
        B = np.zeros(d + 1)

        # Construct polynomial basis
        for j in range(d + 1):
            # Construct Lagrange polynomials to get the polynomial basis at
            # the collocation point
            p = np.poly1d([1])
            for r in range(d + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # Evaluate the polynomial at the final time to get the
            # coefficients of the continuity equation
            D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation
            # points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(d + 1):
                C[j, r] = pder(tau_root[r])

            # Evaluate the integral of the polynomial to get the coefficients
            # of the quadrature function
            pint = np.polyint(p)
            B[j] = pint(1.0)

        return CollocationMatrices(
            order=d,
            root=tau_root,
            B=B,
            C=C,
            D=D,
        )


class CasADiMHEBackend(CasADiBackend):
    """
    Class doing optimization of ADMM subproblems with CasADi.
    """

    system_type = MHESystem
    discretization_types = {
        DiscretizationMethod.collocation: DirectCollocation,
    }
    system: MHESystem

    @staticmethod
    def sample(
        trajectory: Union[float, int, pd.Series, list[Union[float, int]]],
        grid: Union[list, np.ndarray],
        current: float = 0,
        method: str = "linear",
    ) -> list:
        """
        Obtain the specified portion of the trajectory.

        Args:
            trajectory:  The trajectory to be sampled. Scalars will be
                expanded onto the grid. Lists need to exactly match the provided
                grid. Otherwise, a list of tuples is accepted with the form (
                timestamp, value). A dict with the keys 'grid' and 'value' is also
                accepted.
            current: start time of requested trajectory
            grid: target interpolation grid in seconds in relative terms (i.e.
                starting from 0 usually)
            method: interpolation method, currently accepted: 'linear',
                'spline', 'previous'

        Returns:
            Sampled list of values.

        Takes a slice of the trajectory from the current time step with the
        specified length and interpolates it to match the requested sampling.
        If the requested horizon is longer than the available data, the last
        available value will be used for the remainder.

        Raises:
            ValueError
            TypeError
        """
        target_grid_length = len(grid)
        if isinstance(trajectory, (float, int)):
            # return constant trajectory for scalars
            return [trajectory] * target_grid_length
        if isinstance(trajectory, list):
            # return lists of matching length without timestamps
            if len(trajectory) == target_grid_length:
                return trajectory
            raise ValueError(
                f"Passed list with length {len(trajectory)} "
                f"does not match target ({target_grid_length})."
            )
        if isinstance(trajectory, pd.Series):
            source_grid = np.array(trajectory.index)
            values = trajectory.values
        else:
            raise TypeError(
                f"Passed trajectory of type '{type(trajectory)}' " f"cannot be sampled."
            )
        target_grid = np.array(grid) + current

        # expand scalar values
        if len(source_grid) == 1:
            return [trajectory[0]] * target_grid_length

        # skip resampling if grids are (almost) the same
        if (target_grid.shape == source_grid.shape) and all(target_grid == source_grid):
            return list(values)
        values = np.array(values)

        # check requested portion of trajectory, whether the most recent value in the
        # source grid is older than the first value in the MHE trajectory
        if target_grid[0] >= source_grid[-1]:
            # return the last value of the trajectory if requested sample
            # starts out of range
            logger.warning(
                f"Latest value of source grid %s is older than "
                f"current time (%s. Returning latest value anyway.",
                source_grid[-1],
                current,
            )
            return [values[-1]] * target_grid_length

        # determine whether the target grid lies within the available source grid, and
        # how many entries to extrapolate on either side
        source_grid_oldest_time: float = source_grid[0]
        source_grid_newest_time: float = source_grid[-1]
        source_is_recent_enough: np.ndarray = target_grid < source_grid_newest_time
        source_is_old_enough: np.ndarray = target_grid > source_grid_oldest_time
        number_of_missing_old_entries: int = target_grid_length - np.count_nonzero(
            source_is_old_enough
        )
        number_of_missing_new_entries: int = target_grid_length - np.count_nonzero(
            source_is_recent_enough
        )
        # shorten target interpolation grid by extra points that go above or below
        # available data range
        target_grid = target_grid[source_is_recent_enough * source_is_old_enough]

        # interpolate data to match new grid
        if method == "linear":
            tck = interpolate.interp1d(x=source_grid, y=values, kind="linear")
            sequence_new = list(tck(target_grid))
        elif method == "spline":
            raise NotImplementedError(
                "Spline interpolation is currently not " "supported"
            )
        elif method == "previous":
            tck = interpolate.interp1d(source_grid, values, kind="previous")
            sequence_new = list(tck(target_grid))
        else:
            raise ValueError(
                f"Chosen 'method' {method} is not a valid method. "
                f"Currently supported: linear, spline, previous"
            )

        # extrapolate sequence with last available value if necessary
        interpolated_trajectory = (
            [values[0]] * number_of_missing_old_entries
            + sequence_new
            + [values[-1]] * number_of_missing_new_entries
        )

        return interpolated_trajectory
