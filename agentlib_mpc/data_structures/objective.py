import pandas as pd
import numpy as np
import re
import casadi as ca
from typing import Union
from agentlib_mpc.models.casadi_model import CasadiParameter, CasadiInput


def is_casadi_expression(obj) -> bool:
    """Checks if the object is specifically a CasADi expression (not a pure, symbolic casadi variable)"""
    return isinstance(obj, (ca.MX, ca.SX)) and not obj.is_symbolic()


class SubObjective:
    def __init__(
        self,
        expressions: ca.MX,
        weight: Union[float, int, CasadiParameter] = 1,
        name: str = None,
    ):
        """
        Create an objective term

        Args:
            expressions: Expression to be used in the objective
            weight: Weight factor for this objective
            name: Optional name for identification
        """
        self.expression = expressions
        self.weight = weight
        self.name = name or f"obj_{id(self)}"

    def get_weighted_expression(self):
        """Returns the final weighted expression"""
        if is_casadi_expression(self.weight):
            raise ValueError(
                f"Objective {self.name} contains expression {self.weight} in weight but only "
                f"original variables are allowed."
            )
        return self.weight * self.expression

    def calculate_value(self, data, weight):
        """Calculate the objective value from data"""
        ts = np.diff(data.index)
        result = self._evaluate_expression(self.expression, data)
        return sum(weight * result * ts)

    def _evaluate_expression(self, expr, df):
        """Evaluate a complex expression using dataframe values. This function
        recreates the computation for the objective values from the string
        representation of the casadi expression. In future versions we might use the
        direct expression with a casadi function and map from the available variables"""
        # Handle simple named variables first
        var_name = expr.name
        for col_type in ["variable", "parameter"]:
            if (col_type, var_name) in df.columns:
                return df.loc[:, (col_type, var_name)].values[:-1]

        expr_str = str(expr)

        # Handle common CasADi functions with simple replacements
        casadi_replacements = {
            "sq(": "(",
            "fabs(": "abs(",
            "sqrt(": "sqrt(",
            "sin(": "sin(",
            "cos(": "cos(",
            "exp(": "exp(",
            "log(": "log(",
        }

        # Apply replacements
        eval_str = expr_str
        is_square = False
        if "sq(" in eval_str:
            is_square = True
            eval_str = eval_str.replace("sq(", "(")

        for casadi_func, replacement in casadi_replacements.items():
            if casadi_func != "sq(":  # already handled above
                eval_str = eval_str.replace(casadi_func, replacement)

        # Extract variable names, filtering out CasADi function names
        casadi_functions = [
            "sq",
            "fabs",
            "sqrt",
            "sin",
            "cos",
            "exp",
            "log",
            "abs",
            "max",
            "min",
        ]
        var_names = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", expr_str)
        var_names = [name for name in var_names if name not in casadi_functions]

        if not var_names:
            try:
                const_val = float(expr_str.strip("()"))
                return np.full(len(df.index) - 1, const_val)
            except:
                pass

        values_found = {}
        for var_name in var_names:
            for col_type in ["variable", "parameter"]:
                if (col_type, var_name) in df.columns:
                    values_found[var_name] = df.loc[:, (col_type, var_name)].values[:-1]
                    break

        if not values_found:
            raise ValueError(f"No variables found in dataframe for expression: {expr}")

        try:
            safe_dict = values_found.copy()

            # Handle common mathematical operations and numpy functions
            safe_dict.update(
                {
                    "abs": np.abs,
                    "sqrt": np.sqrt,
                    "sin": np.sin,
                    "cos": np.cos,
                    "exp": np.exp,
                    "log": np.log,
                    "max": np.maximum,
                    "min": np.minimum,
                }
            )

            # Remove outer parentheses if they wrap the entire expression
            if eval_str.startswith("(") and eval_str.endswith(")"):
                paren_count = 0
                is_outermost = True
                for i, char in enumerate(eval_str[1:-1], 1):
                    if char == "(":
                        paren_count += 1
                    elif char == ")":
                        paren_count -= 1
                        if paren_count < 0:
                            is_outermost = False
                            break
                if is_outermost and paren_count == 0:
                    eval_str = eval_str[1:-1]

            result = eval(eval_str, {"__builtins__": {}}, safe_dict)

            # Apply square if it was sq() function
            if is_square:
                result = result**2

            if isinstance(result, np.ndarray):
                return result
            elif isinstance(result, (int, float)):
                return np.full(len(df.index) - 1, result)
            else:
                return np.array(result)

        except Exception as e:
            # If evaluation fails, try simple pattern matching as fallback
            if len(values_found) == 1:
                var_name = list(values_found.keys())[0]
                if expr_str.startswith("(-") or expr_str.startswith("-"):
                    return -values_found[var_name]
                elif is_square:
                    return values_found[var_name] ** 2
                else:
                    return values_found[var_name]

            raise ValueError(f"Unable to evaluate expression: {expr}. Error: {e}")


class ChangePenaltyObjective(SubObjective):
    def __init__(
        self,
        expressions: CasadiInput,
        weight: Union[float, int, CasadiParameter],
        name: str = None,
    ):
        """
        Args:
            expressions: Control variable to track changes
            weight: Weight factor for this objective
            name: Optional name for identification/reporting
        """
        self.control: CasadiInput = expressions
        if not isinstance(expressions, CasadiInput):
            raise TypeError(
                "Tried to create a control change objective with an "
                "expression or different type of CasadiVariable. "
                "Currently, only raw CasadiInputs are supported."
            )
        super().__init__(
            expressions=expressions,
            weight=weight,
            name=name or f"delta_{self.get_control_name()}",
        )

    def get_control_name(self):
        """Return the name of the associated control variable"""
        return self.control.name

    def get_weighted_expression(self):
        """
        Override parent method to provide a placeholder.
        The actual penalty calculation happens in the discretization step.
        """
        return 0

    def calculate_value(self, series, weight):
        """Returns the final weighted result by multiplying all expressions"""
        diff_values = series.diff()
        diff = diff_values.values[1:]
        ts = np.diff(series.index)
        results = weight**2 * diff**2 * ts
        return sum(results.dropna())


class CombinedObjective:
    """Container for multiple objective terms with normalization"""

    def __init__(self, *objectives, normalization: float = 1.0):
        """
        Args:
            *objectives: Variable number of objective terms
            normalization: Global normalization factor
        """
        self.objectives = list(objectives)
        self.normalization = normalization
        self._values = {}

    def get_delta_u_objectives(self):
        """Returns a list of all ChangePenaltyObjective instances"""
        return [obj for obj in self.objectives if isinstance(obj, ChangePenaltyObjective)]

    @property
    def expression(self):
        return self.get_casadi_expression()

    def get_casadi_expression(self):
        """Combine all objectives into a single CasADi expression"""
        terms = []
        for obj in self.objectives:
            terms.append(obj.get_weighted_expression())
        if terms:
            return sum(terms) / self.normalization
        return 0

    def calculate_values(self, result_df, grid):
        """Calculate values for each objective component using the result dataframe"""
        self._values = {}
        df = self._prepare_dataframe(result_df, grid)
        total_value = 0

        for obj in self.objectives:
            name = obj.name
            # Handle symbolic or numeric weights
            if hasattr(obj.weight, "sym"):
                weight_name = obj.weight.name
                if ("parameter", weight_name) in df.columns:
                    weight = df.loc[:, ("parameter", weight_name)].iloc[:-1]
                else:
                    weight = obj.weight
            else:
                weight = obj.weight
            if isinstance(obj, ChangePenaltyObjective):
                control_name = obj.get_control_name()
                control_series = df.loc[:, ("variable", control_name)]
                value = obj.calculate_value(control_series, weight)
                self._values[name] = value / self.normalization
            elif isinstance(obj, SubObjective):
                value = obj.calculate_value(df, weight)
                self._values[name] = value / self.normalization
            if self._values[name] is not None:
                total_value += self._values[name]

        self._values["total"] = total_value
        return self._values

    def _prepare_dataframe(self, df, grid=None):
        """
        Convert DataFrame index to numeric values and handle NaN values for calculation.

        Args:
            df: DataFrame with potentially string tuple indices
            grid: Optional list of indices to consider in the result

        Returns:
            DataFrame with numeric index and processed values
            The time value from the first element of tuple index
        """
        new_df = df.copy()

        if len(df.index) > 0 and isinstance(df.index[0], str) and "(" in df.index[0]:
            new_indices = []
            for idx in df.index:
                try:
                    parts = idx.strip("()").split(",", 1)
                    new_indices.append(float(parts[1].strip()))
                except (ValueError, IndexError):
                    new_indices.append(idx)
            new_df.index = new_indices

        for col in new_df.columns:
            if col[0] in ["upper", "lower"]:
                continue
            if col[0] == "parameter":
                new_df[col] = new_df[col].ffill()
            elif col[0] == "variable":
                self._handle_nan_values(new_df, col, grid)

        if grid is not None and len(grid) > 0:
            valid_grid = [g for g in grid if g in new_df.index]
            if valid_grid:
                new_df = new_df.loc[valid_grid]

        return new_df

    def _handle_nan_values(self, df, col, grid=None):
        series = df[col]

        if grid is not None and len(grid) > 0:
            grid_values = [v for v in grid if v in df.index]
            grid_series = series.loc[grid_values]

            if grid_series.isna().all():
                self._fill_collocation_nans(df, col)
                return
        pass

    def _fill_collocation_nans(self, df, col):
        series = df[col]
        new_series = series.copy()
        nan_indices = np.where(series.isna())[0]

        for i in range(len(nan_indices)):
            nan_idx = nan_indices[i]
            next_values = []
            j = nan_idx + 1
            while j < len(series) and not pd.isna(series.iloc[j]):
                next_values.append(series.iloc[j])
                j += 1
            if next_values:
                mean_val = sum(next_values) / len(next_values)
                new_series.iloc[nan_idx] = mean_val
        df[col] = new_series


class ConditionalObjective:
    """Represents a conditional objective that switches between different objectives based on conditions"""

    def __init__(self, *condition_objective_pairs, default_objective=None):
        """
        Args:
            *condition_objective_pairs: Tuples of (condition, objective)
                where condition is a CasADi expression that evaluates to True/False
                and objective is a CombinedObjective
            default_objective: The objective to use when all conditions are False
        """
        self.condition_objective_pairs = condition_objective_pairs
        self.default_objective = default_objective or CombinedObjective()

        self.all_objectives = [self.default_objective]
        for _, objective in condition_objective_pairs:
            if objective not in self.all_objectives:
                self.all_objectives.append(objective)

        self._flattened_objectives = []
        for obj in self.all_objectives:
            if hasattr(obj, "objectives"):
                self._flattened_objectives.extend(obj.objectives)

    @property
    def objectives(self):
        """Return flattened list of all objective terms for reporting"""
        return self._flattened_objectives

    def get_casadi_expression(self):
        """Combine all objectives into a conditional CasADi expression"""
        result = self.default_objective.get_casadi_expression()

        for condition, objective in reversed(self.condition_objective_pairs):
            result = ca.if_else(condition, objective.get_casadi_expression(), result)

        return result

    def get_delta_u_objectives(self):
        """Returns all ChangePenaltyObjective instances from all contained objectives"""
        all_delta_u = []
        for objective in self.all_objectives:
            all_delta_u.extend(objective.get_delta_u_objectives())
        return list(set(all_delta_u))

    def calculate_values(self, result_df, grid):
        """
        Calculate values for each objective component based on when conditions are active.
        """
        all_values = {}
        total_value = 0

        df = self.default_objective._prepare_dataframe(result_df.copy(), grid)
        active_objectives = self._determine_active_objectives(df)

        for objective, active_mask in active_objectives.items():
            if not np.any(active_mask):
                continue
                
            active_df = df.loc[active_mask].copy()
            if len(active_df) > 0:
                obj_values = objective.calculate_values(active_df, None)
                for name, value in obj_values.items():
                    if name == "total":
                        continue
                    if name not in all_values:
                        all_values[name] = 0
                    all_values[name] += value
                    if value is not None:
                        total_value += value

        all_values["total"] = total_value
        return all_values

    def _determine_active_objectives(self, df):
        """
        Determine which objective is active at each time step.

        Args:
            df: DataFrame with results

        Returns:
            Dict mapping objectives to boolean masks
        """
        active_objectives = {}

        active_objectives[self.default_objective] = pd.Series(True, index=df.index)

        for _, objective in self.condition_objective_pairs:
            active_objectives[objective] = pd.Series(False, index=df.index)

        for condition, objective in self.condition_objective_pairs:
            condition_mask = self._evaluate_condition(condition, df)

            active_objectives[objective] = condition_mask

            active_objectives[self.default_objective] = active_objectives[self.default_objective] & ~condition_mask

        return active_objectives

    def _evaluate_condition(self, condition, df):
        """
        Evaluate a condition for all rows in the dataframe.

        Args:
            condition: CasADi expression representing the condition
            df: DataFrame with results

        Returns:
            Boolean Series with True where condition is true
        """
        condition_str = str(condition)

        identifier_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*'
        potential_vars = re.findall(identifier_pattern, condition_str)

        operators_keywords = {'and', 'or', 'not', 'in', 'is', 'if', 'else', 'elif', 'for', 'while', 'def', 'class'}
        var_names = [var for var in potential_vars if var not in operators_keywords]

        var_names = list(dict.fromkeys(var_names))

        values_dict = {}

        for var_name in var_names:
            if var_name == "time":
                values_dict["time"] = df.index.to_numpy()
                continue

            found = False
            for col_type in ["variable", "parameter"]:
                if (col_type, var_name) in df.columns:
                    values_dict[var_name] = df.loc[:, (col_type, var_name)].values
                    found = True
                    break

            if not found:
                try:
                    float(var_name)
                except ValueError:
                    print(f"Warning: Variable '{var_name}' not found in DataFrame")
                    values_dict[var_name] = np.zeros(len(df))

        n_rows = len(df)
        mask = np.zeros(n_rows, dtype=bool)

        for i in range(n_rows):
            local_vars = {}
            for var_name, values in values_dict.items():
                if isinstance(values, np.ndarray):
                    local_vars[var_name] = values[i]
                else:
                    local_vars[var_name] = values

            try:
                eval_str = condition_str

                replacements = {
                }

                for casadi_op, python_op in replacements.items():
                    eval_str = eval_str.replace(casadi_op, python_op)

                result = eval(eval_str, {"__builtins__": {}, "abs": abs, "min": min, "max": max}, local_vars)
                mask[i] = bool(result)

            except Exception as e:
                print(f"Error evaluating condition at row {i}: {e}")
                mask[i] = False

        return pd.Series(mask, index=df.index)
