"""Holds classes that implement different transcriptions of the OCP"""
import abc
import dataclasses
from functools import cached_property
from pathlib import Path
from typing import TypeVar, Union, Callable, Optional

import casadi as ca
import numpy as np
import pandas as pd

from agentlib_mpc.data_structures.casadi_utils import (
    CaFuncInputs,
    OptVarMXContainer,
    OptParMXContainer,
    CasadiDiscretizationOptions,
    SolverFactory,
    MPCInputs,
    GUESS_PREFIX,
)
from agentlib_mpc.optimization_backends.casadi_.core.VariableGroup import (
    OptimizationQuantity,
    OptimizationParameter,
    OptimizationVariable,
)
from agentlib_mpc.optimization_backends.casadi_.core.system import System


CasadiVariableList = Union[list[ca.MX], ca.MX]


@dataclasses.dataclass
class Results:
    matrix: ca.MX
    grid: list[float]
    columns: pd.MultiIndex
    stats: dict
    variable_grid_indices: dict[str, list[int]]
    _variable_name_to_index: dict[str, int] = None

    def __post_init__(self):
        self._variable_name_to_index = self.variable_lookup()
        try:
            iters = self.stats.pop("iterations")
            self.stats["obj"] = iters["obj"][-1]
        except KeyError:
            pass
        if "fatrop" in self.stats:
            self.stats.pop("ng")
            self.stats.pop("nu")
            self.stats.pop("nx")
            self.stats.pop("fatrop")

    def __getitem__(self, item: str) -> np.ndarray:
        return self.matrix[
            self.variable_grid_indices[item], self._variable_name_to_index[item]
        ].toarray(simplify=True)

    def variable_lookup(self) -> dict[str, int]:
        """Creates a mapping from variable names to the column index in the Matrix"""
        lookup = {}
        for index, label in enumerate(self.columns):
            if label[0] == "variable":
                lookup[label[1]] = index
        return lookup

    @cached_property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix, index=self.grid, columns=self.columns)

    def write_columns(self, file: Path):
        df = pd.DataFrame(columns=self.columns)
        df.to_csv(file)

    def write_stats_columns(self, file: Path):
        line = f""",{",".join(self.stats)}\n"""
        with open(file, "w") as f:
            f.write(line)

    def stats_line(self, index: str) -> str:
        return f""""{index}",{",".join(map(str, self.stats.values()))}\n"""


class Discretization(abc.ABC):
    """
    opt_vars: holds symbolic variables during problem creation
    opt_vars_lb: holds symbolic variables during problem creation
    opt_vars_ub: holds symbolic variables during problem creation
    initial_guess: holds symbolic variables during problem creation
    opt_pars: holds symbolic variables during problem creation
    constraints: holds symbolic variables during problem creation
    constraints_lb: holds symbolic variables during problem creation
    constraints_ub: holds symbolic variables during problem creation
    objective_function: cost function during problem creation
    mpc_opt_vars (dict): holds the symbolic variables and grids during
        problem creation sorted by type as in system_variables
    mpc_opt_pars (dict): holds the symbolic variables and grids during
        problem creation sorted by type as in system_parameters
    """

    _stage_function: ca.Function
    _mpc_inputs_to_nlp_inputs: ca.Function
    _nlp_outputs_to_mpc_outputs: ca.Function
    _optimizer: ca.Function
    _result_map: ca.Function

    def __init__(self, options: CasadiDiscretizationOptions):
        self.options = options
        self._finished_discretization: bool = False

        # attributes used for problem creation
        self.k: int = 0  # increment for prediction loop
        self.pred_time: float = 0  # for creation of grids

        # lists that hold all variables of the optimization problem
        self.opt_vars: CasadiVariableList = []  # hold all optimization variables
        self.opt_vars_lb: list[ca.MX] = []
        self.opt_vars_ub: list[ca.MX] = []
        self.initial_guess: list[ca.MX] = []
        self.opt_pars: CasadiVariableList = []  # hold all optimization parameters
        self.constraints: CasadiVariableList = []
        self.constraints_lb: list[ca.MX] = []
        self.constraints_ub: list[ca.MX] = []
        self.objective_function: CaFuncInputs = ca.DM(0)
        self.binary_opt_vars = []
        self.equalities: list[bool] = []

        # dicts of variables of the optimization problem, sorted by role
        self.mpc_opt_vars: dict[str, OptVarMXContainer] = {}
        self.mpc_opt_pars: dict[str, OptParMXContainer] = {}

        self._create_results: Optional[Callable[[ca.DM, dict], Results]] = None

    def initialize(self, system: System, solver_factory: SolverFactory):
        """Initializes the trajectory optimization problem, creating all symbolic
        variables of the OCP, the mapping function and the numerical solver."""
        self._discretize(system)
        self._finished_discretization = True
        self.create_nlp_in_out_mapping(system)
        self._create_solver(solver_factory)

    @abc.abstractmethod
    def _discretize(self, sys: System):
        """Specifies the discretization of direct optimization methods like
        collocation, multiple shooting etc. This function creates the lists of
        variables, parameters, constraints etc. by using the self.add_opt_var functions.
        For an example see optimization_backends.casadi_.basic
        """
        ...

    def _create_solver(self, solver_factory: SolverFactory):
        self._optimizer = solver_factory.create_solver(
            nlp=self.nlp, discrete=self.binary_vars, equalities=self.equalities
        )

    def solve(self, mpc_inputs: MPCInputs) -> Results:
        """
        Solves the discretized trajectory optimization problem.

        Args:
            mpc_inputs: Casadi Matrices specifying the input of all different types
                of optimization parameters. Matrices consist of different variable rows
                and have a column for each time step in the discretization.
                There are separate matrices for each input type (as defined in the
                System), and also for the upper and lower boundaries of variables
                respectively.


        Returns:
            Results: The complete evolution of the states, inputs and boundaries of each
                variable and parameter over the prediction horizon, as well as solve
                statistics.

        """
        # collect and format inputs
        guesses = self._determine_initial_guess(mpc_inputs)
        mpc_inputs.update(guesses)
        nlp_inputs: dict[str, ca.DM] = self._mpc_inputs_to_nlp_inputs(**mpc_inputs)

        # perform optimization
        nlp_output = self._optimizer(**nlp_inputs)

        # format and return solution
        mpc_output = self._nlp_outputs_to_mpc_outputs(vars_at_optimum=nlp_output["x"])
        self._remember_solution(mpc_output)
        result = self._process_solution(inputs=mpc_inputs, outputs=mpc_output)
        return result

    def _determine_initial_guess(self, mpc_inputs: MPCInputs) -> MPCInputs:
        """
        Collects initial guesses for all mpc variables. If possible, uses result
        of last optimization.
        If not available, the current measurement is used for states, and the mean of
        the upper and lower bound is used otherwise.
        """
        guesses = {}

        for denotation, var in self.mpc_opt_vars.items():
            guess = var.opt
            if guess is None:
                # if initial value is available, assume it is constant and make guess
                guess_denotation = f"initial_{denotation}"
                if guess_denotation in mpc_inputs:
                    # changes here because of the long guess.array caused by np.tile for lags
                    if mpc_inputs[guess_denotation].shape[1] > 1:
                        state_measurements = mpc_inputs[guess_denotation][:, -1]
                    else:
                        state_measurements = mpc_inputs[guess_denotation]
                    guess = np.tile(state_measurements, len(var.grid))
                # get guess from boundaries if last optimum is not available
                else:
                    guess = np.array(
                        0.5
                        * (
                            mpc_inputs[f"lb_{denotation}"]
                            + mpc_inputs[f"ub_{denotation}"]
                        )
                    )
                    guess = np.nan_to_num(
                        guess, posinf=100_000_000, neginf=-100_000_000
                    )
            guesses.update({GUESS_PREFIX + denotation: guess})

        return guesses

    def _remember_solution(self, optimum: dict[str, ca.DM]):
        """Saves the last optimal solution for all optimization variables
        sorted by type."""
        for den, var in self.mpc_opt_vars.items():
            var.opt = optimum[den]

    def _process_solution(self, inputs: dict, outputs: dict) -> Results:
        """
        If self.result_file is not empty,
        collect all inputs and outputs of the optimization problem and format
        them as DataFrames and pass them to OptimizationBackend.save_df().
        Args:
            inputs: mpc_inputs dict returned from _get_current_mpc_inputs
            outputs: mpc_output from self._nlp_outputs_to_mpc_outputs
        """
        # update the guess values at the variable positions with the outputs
        for key, value in inputs.items():
            key: str
            if key.startswith(GUESS_PREFIX):
                out_key = key[len(GUESS_PREFIX) :]
                inputs[key] = outputs[out_key]

        result_matrix = self._result_map(**inputs)["result"]



        return self._create_results(result_matrix, self._optimizer.stats())

    def create_nlp_in_out_mapping(self, system: System):
        """
        Function creating mapping functions between the MPC variables ordered
        by type (as defined in `declare_quantities` and the raw input/output
        vector of the CasADi NLP.
        """
        # Concatenate nlp variables to CasADi MX vectors
        self.opt_vars = ca.vertcat(*self.opt_vars)
        self.constraints = ca.vertcat(*self.constraints)
        self.opt_pars = ca.vertcat(*self.opt_pars)
        initial_guess = ca.vertcat(*self.initial_guess)
        opt_vars_lb = ca.vertcat(*self.opt_vars_lb)
        opt_vars_ub = ca.vertcat(*self.opt_vars_ub)
        constraints_lb = ca.vertcat(*self.constraints_lb)
        constraints_ub = ca.vertcat(*self.constraints_ub)

        # nlp inputs
        nlp_inputs = [
            self.opt_pars,
            initial_guess,
            opt_vars_lb,
            opt_vars_ub,
            constraints_lb,
            constraints_ub,
        ]
        nlp_input_denotations = [
            "p",
            "x0",
            "lbx",
            "ubx",
            "lbg",
            "ubg",
        ]

        # create empty lists to store all nlp inputs and outputs
        mpc_inputs = []
        mpc_input_denotations = []
        mpc_outputs = []
        mpc_output_denotations = []

        # Concatenate mpc outputs and their bounds to CasADi MX matrices
        for denotation, opt_var in self.mpc_opt_vars.items():
            # mpc opt vars
            var = opt_var.var
            var = ca.horzcat(*var)
            mpc_outputs.append(var)
            mpc_output_denotations.append(denotation)

            # their bounds and guess
            lb = ca.horzcat(*opt_var.lb)
            ub = ca.horzcat(*opt_var.ub)
            guess = ca.horzcat(*opt_var.guess)
            mpc_inputs.extend([lb, ub, guess])
            mpc_input_denotations.extend(
                [f"lb_{denotation}", f"ub_{denotation}", GUESS_PREFIX + denotation]
            )

        # Concatenate mpc inputs to CasADi MX matrices
        for denotation, opt_par in self.mpc_opt_pars.items():
            var = opt_par.var
            var = ca.horzcat(*var)
            mpc_inputs.append(var)
            mpc_input_denotations.append(denotation)

        # Mapping function that rearranges the variables for input into the NLP
        self._mpc_inputs_to_nlp_inputs = ca.Function(
            "mpc_inputs_to_nlp_inputs",
            mpc_inputs,
            nlp_inputs,
            mpc_input_denotations,
            nlp_input_denotations,
        )

        # Mapping function that rearranges the output of the nlp and sorts
        # by denotation
        self._nlp_outputs_to_mpc_outputs = ca.Function(
            "nlp_outputs_to_mpc_outputs",
            [self.opt_vars],
            mpc_outputs,
            ["vars_at_optimum"],
            mpc_output_denotations,
        )

        matrix, col_index, full_grid, var_grids = self._create_result_format(system)
        self._result_map = ca.Function(
            "result_map", mpc_inputs, [matrix], mpc_input_denotations, ["result"]
        )

        def make_results_view(result_matrix: ca.DM, stats: dict) -> Results:
            return Results(
                matrix=result_matrix,
                columns=col_index,
                grid=full_grid,
                variable_grid_indices=var_grids,
                stats=stats,
            )

        self._create_results = make_results_view

    @property
    def nlp(self) -> dict[str, ca.MX]:
        """The nlp dict that casadi solvers need for instantiation"""
        if not self._finished_discretization:
            raise RuntimeError("You have to initialize first")
        return {
            "x": self.opt_vars,
            "f": self.objective_function,
            "g": self.constraints,
            "p": self.opt_pars,
        }

    @property
    def binary_vars(self) -> list[bool]:
        """List specifying for every optimization variable, whether it is binary."""
        if not self._finished_discretization:
            raise RuntimeError("You have to initialize first")
        return self.binary_opt_vars

    def _create_result_format(
        self, system: System
    ) -> (ca.MX, pd.MultiIndex, list[float], dict[str, list[int]]):
        """
        Creates an MX matrix that includes all inputs and outputs of the nlp
        in an ordered format.
        Sets the _result_columns and _full_index private attributes.

        Created format:
                variable upper lower parameter ...
                t_0      t_0   t_0   rho       ...
        time    .        .      .     .
        1       .        .      .     .
        2       .        .      .     .
        3
        4
        5
        6

        Returns:
            The Matrix as MX that defines the output format of the solver
            A pandas column index for this matrix
            The full grid
            A dict specifying the row index with non-nan values for all variables
                (not parameters)

        """

        def make_column(vars_in_quantity: int, grid: list) -> (ca.MX, list[int]):
            """Creates a matrix with the width of the number of variables in a quantity
            group, and the length of the full grid. Also returns the indexes of this
             variable group that point to non-nan entries. The indices are used in
             slices of the results object."""
            col = []
            non_nan_entries = []
            for index_full, time in enumerate(full_grid):
                if time in grid:
                    index = grid.index(time)
                    entry = mx_list[index].T
                    if time >= 0:
                        # with NARX there can be times smaller 0, however we don't
                        # want them in the results slice
                        non_nan_entries.append(index_full)
                else:
                    entry = np.full((1, vars_in_quantity), np.nan)
                col = ca.vertcat(col, entry)

            return col, non_nan_entries

        full_grid = set()
        variable_grids: dict[str, list[int]] = {}
        for quant_type in {**self.mpc_opt_vars, **self.mpc_opt_pars}.values():

            full_grid.update(set(quant_type.grid))
        full_grid = sorted(full_grid)
        columns = []
        output_matrix = ca.MX.sym("Results", len(full_grid), 0)

        for sys_pars in system.parameters:
            names_list = sys_pars.full_names
            if not names_list:
                continue

            columns.extend(list(map(lambda x: ("parameter", x), names_list)))
            grid = self.grid(sys_pars)
            mx_list = self.mpc_opt_pars[sys_pars.name].var
            column, _ = make_column(len(names_list), grid)
            output_matrix = ca.horzcat(output_matrix, column)

        for sys_vars in system.variables:
            names_list = sys_vars.full_names

            if not names_list:
                continue

            grid = self.grid(sys_vars)
            iterator = [("var", "variable"), ("ub", "upper"), ("lb", "lower")]
            for key, header in iterator:
                columns.extend(list(map(lambda x: (header, x), names_list)))
                mx_list = self.mpc_opt_vars[sys_vars.name].__dict__[key]
                column, grid_indices = make_column(len(names_list), grid)
                output_matrix = ca.horzcat(output_matrix, column)

                if key == "var":
                    variable_grids.update({n: grid_indices for n in names_list})

        result_columns = pd.MultiIndex.from_tuples(columns)
        return output_matrix, result_columns, full_grid, variable_grids

    def add_opt_var(
        self,
        quantity: OptimizationVariable,
        lb: ca.MX = None,
        ub: ca.MX = None,
        guess: float = None,
        post_den: str = "",
    ):
        """
        Create an optimization variable and append to all the associated
        lists. If lb or ub are given, they override the values provided at
        runtime! The usual application of this is, to fix the initial value
        of a state to a parameter.

        Args:
            quantity: corresponding system variable
            lb: lower bound of the variable
            ub: upper bound of the variable
            guess: default for the initial guess
            post_den: string to add to casadi MX after denotation (for debugging)
        """
        # get dimension
        dimension = quantity.dim
        denotation = quantity.name

        # create symbolic variables
        opt_var = ca.MX.sym(f"{denotation}_{self.pred_time}{post_den}", dimension)
        lower = ca.MX.sym(f"lb_{denotation}_{self.pred_time}{post_den}", dimension)
        upper = ca.MX.sym(f"ub_{denotation}_{self.pred_time}{post_den}", dimension)

        # if are not given (generally true), use the default variable for nlp lists
        if lb is None:
            lb = lower
        if ub is None:
            ub = upper

        # append to nlp specific lists
        self.opt_vars.append(opt_var)
        self.opt_vars_lb.append(lb)
        self.opt_vars_ub.append(ub)
        if guess is None:
            guess = opt_var
        self.initial_guess.append(guess)
        self.binary_opt_vars.extend([quantity.binary] * dimension)

        # append to variable specific lists
        var_list = self.mpc_opt_vars.setdefault(denotation, OptVarMXContainer())
        var_list.var.append(opt_var)
        var_list.lb.append(lower)
        var_list.ub.append(upper)
        var_list.guess.append(opt_var)
        var_list.grid.append(self.pred_time)

        return opt_var

    def add_opt_par(self, quantity: OptimizationParameter, post_den: str = ""):
        """
        Create an optimization parameter and append to all the associated lists.

        denotation[str]: the key of the parameter, e.g. 'P', 'Q', ...
        dimension[int]: the dimension of the parameter
        post_den[str]: string to add to casadi MX after denotation (for debugging)
        """
        # get dimension
        dimension = quantity.dim
        denotation = quantity.name

        # create symbolic variables
        opt_par = ca.MX.sym(f"{denotation}_{self.pred_time}{post_den}", dimension)
        self.opt_pars.append(opt_par)

        # append to variable specific lists
        par_list = self.mpc_opt_pars.setdefault(denotation, OptParMXContainer())
        par_list.var.append(opt_par)
        par_list.grid.append(self.pred_time)

        return opt_par

    def add_constraint(
        self,
        constraint_function: CaFuncInputs,
        lb: CaFuncInputs = None,
        ub: CaFuncInputs = None,
            *,
            gap_closing: bool = False,
    ):
        """
        Add a constraint to the optimization problem. If no bounds are given,
        adds an equality constraint.
        """
        # set equality for fatrop
        self.equalities.extend([gap_closing]*constraint_function.shape[0])

        # set bounds to default for equality constraints
        if lb is None:
            lb = ca.DM.zeros(constraint_function.shape[0], 1)
        if ub is None:
            ub = ca.DM.zeros(constraint_function.shape[0], 1)

        # Append inequality constraints
        self.constraints.append(constraint_function)
        self.constraints_lb.append(lb)
        self.constraints_ub.append(ub)

    def grid(self, var: OptimizationQuantity) -> list[float]:
        denotation = var.name
        if isinstance(var, OptimizationVariable):
            return self.mpc_opt_vars[denotation].grid
        if isinstance(var, OptimizationParameter):
            return self.mpc_opt_pars[denotation].grid


DiscretizationT = TypeVar("DiscretizationT", bound=Discretization)
