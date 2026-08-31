import pandas as pd
import numpy as np
import re
import casadi as ca
from typing import Union
import warnings
from agentlib_mpc.models.casadi_model import CasadiParameter, CasadiInput


class CompositeWeight:
    def __init__(self, base_component):
        """
        Create a composite weight that tracks original parameter names
        """
        self.param_names = []  # List of parameter names involved
        self.constant_factor = 1.0

        if isinstance(base_component, CasadiParameter):
            self.param_names = [base_component.name]
            self.sym = base_component.sym
        elif isinstance(base_component, (int, float)):
            self.constant_factor = base_component
            self.sym = base_component
        elif isinstance(base_component, CompositeWeight):
            self.param_names = base_component.param_names.copy()
            self.constant_factor = base_component.constant_factor
            self.sym = base_component.sym

        self._update_name()

    def multiply_by(self, other):
        """Multiply this weight by another component"""
        
        if isinstance(other, CasadiParameter):
            self.param_names.append(other.name)
            self.sym *= other.sym
        elif isinstance(other, (int, float)):
            self.constant_factor *= other
            self.sym *= other
        elif isinstance(other, CompositeWeight):
            self.param_names.extend(other.param_names)
            self.constant_factor *= other.constant_factor
            self.sym *= other.sym

        self._update_name()
        return self

    def _update_name(self):
        """Generate a descriptive name based on components"""
        parts = []

        if self.param_names:
            parts.extend(self.param_names)

        if self.constant_factor != 1.0:
            if parts:
                parts.append(f"*{self.constant_factor}")
            else:
                parts.append(str(self.constant_factor))

        self.name = "_times_".join(parts) if parts else "1"

    def evaluate(self, df):
        """Evaluate the composite weight using values from the dataframe"""
        result = self.constant_factor

        for param_name in self.param_names:
            param_values = df.loc[:, ("parameter", param_name)]
            result *= param_values

        return result

def _split_top_level(s, sep=","):
    """
    Split a string on `sep`, but only at nesting depth 0 (i.e. not inside
    parentheses or brackets). Used to parse CasADi's printed expressions,
    where subexpression definitions are comma-separated at the top level
    but may themselves contain commas (e.g. function calls like
    `if_else(a, b, c)`).
    """
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _replace_subexpressions(expr_str, verbose=False):
    """
    Replace CasADi subexpression definitions (marked with @N=...) in an expression string
    with their actual definitions and inline them into the main expression.

    """
    if "@" not in expr_str:
        return expr_str

    # Split into top-level comma-separated segments: each is either a
    # `@N=<definition>` or (for the last segment) the main expression.
    # Splitting at top level only (not inside parentheses) ensures
    # definitions containing commas, e.g. if_else(a, b, c), stay intact.
    parts = _split_top_level(expr_str)

    defs = {}
    remainder_parts = []
    for part in parts:
        part = part.strip()
        match = re.match(r"^(@\d+)=(.*)$", part, re.DOTALL)
        if match:
            defs[match.group(1)] = match.group(2)
        else:
            remainder_parts.append(part)

    clean_expr = ", ".join(remainder_parts)

    # Inline subexpressions into the main expression
    # Process in order to handle nested references (@2 might reference @1),
    # substituting into both the main expression and any remaining
    # (not-yet-resolved) definitions.
    for subexpr in sorted(defs.keys(), key=lambda x: int(x[1:])):
        definition = f"({defs[subexpr]})"
        clean_expr = clean_expr.replace(subexpr, definition)
        for other in defs:
            defs[other] = defs[other].replace(subexpr, definition)

    if verbose:
        print("Subexpressions found in objective expression:")
        for subexpr, definition in defs.items():
            print(f"  {subexpr} = {definition}")
        print(f"Expression before removing subexpressions: {expr_str}")
        print(f"Cleaned expression after removing subexpressions: {clean_expr}")
    
    return clean_expr


def _replace_ternary(eval_str):
    """
    Replace CasADi ternary expressions like (cond)?true:false with where(cond, true, false).
    """
    while "?" in eval_str:
        q_idx = eval_str.rfind("?")
        c_idx = eval_str.find(":", q_idx)
        if q_idx == -1 or c_idx == -1:
            break
            
        # Backward search for condition
        pares = 0
        cond_start = 0
        for i in range(q_idx - 1, -1, -1):
            if eval_str[i] == ")":
                pares += 1
            elif eval_str[i] == "(":
                pares -= 1
                if pares < 0:
                    cond_start = i + 1
                    break
        else:
            cond_start = 0
                    
        # Forward search for false_val
        pares = 0
        false_end = len(eval_str)
        for i in range(c_idx + 1, len(eval_str)):
            if eval_str[i] == "(":
                pares += 1
            elif eval_str[i] == ")":
                pares -= 1
                if pares < 0:
                    false_end = i
                    break
            elif pares == 0 and eval_str[i] in "+-*/,@":
                if i > c_idx + 1:
                    false_end = i
                    break
                    
        cond_str = eval_str[cond_start:q_idx]
        true_str = eval_str[q_idx + 1 : c_idx]
        false_str = eval_str[c_idx + 1 : false_end]
        
        eval_str = (
            eval_str[:cond_start]
            + f"where({cond_str}, {true_str}, {false_str})"
            + eval_str[false_end:]
        )
    return eval_str

def _replace_sq_calls(expr_str):
    """
    Replace CasADi's sq(x) (square) calls with (x)**2, matching each call's
    own closing parenthesis rather than assuming sq(...) spans the whole
    expression - so terms like r*x + s*sq(y) only square the y term.
    """
    while True:
        idx = expr_str.find("sq(")
        if idx == -1:
            break
        start = idx + len("sq(")
        depth = 1
        i = start
        while depth > 0:
            if expr_str[i] == "(":
                depth += 1
            elif expr_str[i] == ")":
                depth -= 1
            i += 1
        inner = expr_str[start : i - 1]
        expr_str = expr_str[:idx] + f"({inner})**2" + expr_str[i:]
    return expr_str


class SubObjective:

    _warned_names = set()

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

    def __add__(self, other):
        """Add two objectives together to create a CombinedObjective"""
        if isinstance(other, SubObjective):
            return CombinedObjective(self, other)
        else:
            raise TypeError(f"Cannot add SubObjective with {type(other)}")

    def __mul__(self, other):
        """Scale objective by a factor"""
        if isinstance(other, (int, float, CasadiParameter)):
            new_weight = self._multiply_weights(self.weight, other)
            return SubObjective(self.expression, new_weight, f"scaled_{self.name}")
        else:
            raise TypeError(f"Cannot multiply SubObjective with {type(other)}")

    def _multiply_weights(self, weight1, weight2):
        """Helper method to properly multiply weights of different types"""
        if isinstance(weight1, (int, float)) and isinstance(weight2, (int, float)):
            return weight1 * weight2

        if isinstance(weight1, (CasadiParameter, CompositeWeight)) or isinstance(weight2,(CasadiParameter, CompositeWeight)):
            result = CompositeWeight(weight1)
            result.multiply_by(weight2)
            return result

        if hasattr(weight1, 'sym'):
            weight1 = weight1.sym
        if hasattr(weight2, 'sym'):
            weight2 = weight2.sym

        return weight1 * weight2

    def get_weighted_expression(self):
        """Returns the final weighted expression"""
        if isinstance(self.weight, (CompositeWeight, CasadiParameter)):
            return self.weight.sym * self.expression
        else:
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

        if "@" in expr_str:
            expr_str = _replace_subexpressions(expr_str)

        if "?" in expr_str:
            expr_str = _replace_ternary(expr_str)
            
        # Replace logical NOT '!' with bitwise NOT '~' for NumPy arrays
        expr_str = re.sub(r"(?<![<>=!])!(?!=)", "~", expr_str)

        # Handle common CasADi functions with simple replacements
        casadi_replacements = {
            "sq(": "(",
            "fabs(": "abs(",
            "sqrt(": "sqrt(",
            "sin(": "sin(",
            "cos(": "cos(",
            "tan(": "tan(",
            "asin(": "arcsin(",
            "acos(": "arccos(",
            "atan(": "arctan(",
            "atan2(": "arctan2(",
            "sinh(": "sinh(",
            "cosh(": "cosh(",
            "tanh(": "tanh(",
            "asinh(": "arcsinh(",
            "acosh(": "arccosh(",
            "atanh(": "arctanh(",
            "exp(": "exp(",
            "log(": "log(",
            "log10(": "log10(",
            "pow(": "power(",
            "floor(": "floor(",
            "ceil(": "ceil(",
            "sign(": "sign(",
            "fmin(": "minimum(",
            "fmax(": "maximum(",
        }

        # Apply replacements
        eval_str = _replace_sq_calls(expr_str)

        for casadi_func, replacement in casadi_replacements.items():
            if casadi_func != "sq(":  # already handled above
                eval_str = eval_str.replace(casadi_func, replacement)

        # Extract variable names, filtering out CasADi function names
        base_funcs = [k.rstrip("(") for k in casadi_replacements.keys()]
        replaced_funcs = [v.rstrip("(") for v in casadi_replacements.values() if v.endswith("(")]
        extra_funcs = ["round", "max", "min", "minimum", "maximum", "where", "abs", "power"]
        casadi_functions = list(set(base_funcs + replaced_funcs + extra_funcs))
        
        var_names = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", expr_str)
        var_names = [name for name in var_names if name not in casadi_functions]

        values_found = {}
        for var_name in var_names:
            for col_type in ["variable", "parameter"]:
                if (col_type, var_name) in df.columns:
                    values_found[var_name] = df.loc[:, (col_type, var_name)].values[:-1]
                    break

        try:
            safe_dict = values_found.copy()

            # Handle common mathematical operations and numpy functions
            safe_dict.update(
                {
                    "abs": np.abs,
                    "sqrt": np.sqrt,
                    "sin": np.sin,
                    "cos": np.cos,
                    "tan": np.tan,
                    "arcsin": np.arcsin,
                    "arccos": np.arccos,
                    "arctan": np.arctan,
                    "arctan2": np.arctan2,
                    "sinh": np.sinh,
                    "cosh": np.cosh,
                    "tanh": np.tanh,
                    "arcsinh": np.arcsinh,
                    "arccosh": np.arccosh,
                    "arctanh": np.arctanh,
                    "exp": np.exp,
                    "log": np.log,
                    "log10": np.log10,
                    "power": np.power,
                    "floor": np.floor,
                    "ceil": np.ceil,
                    "round": np.round,
                    "sign": np.sign,
                    "minimum": np.minimum,
                    "maximum": np.maximum,
                    "max": np.maximum,
                    "min": np.minimum,
                    "where": np.where,
                }
            )

            # Remove outer parentheses if they wrap the entire expression
            if eval_str.startswith("(") and eval_str.endswith(")"):
                eval_str = eval_str[1:-1]

            result = eval(eval_str, {"__builtins__": {}}, safe_dict)

            return result

        except SyntaxError as e:
            if self.name not in SubObjective._warned_names:
                warnings.warn(
                    f"Unable to evaluate expression {self.name} ({expr}). Some terms will be ignored when displaying the objective value. The control still works, only the objective logging is affected.",
                    RuntimeWarning,
                )
                SubObjective._warned_names.add(self.name)
            result = 0
            return result


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

    def __mul__(self, mul):
        """Scale change penalty objective by a factor"""
        if isinstance(mul, (int, float, CasadiParameter)):
            # Scale the weight properly using the parent class method
            new_weight = self._multiply_weights(self.weight, mul)
            scaled_obj = ChangePenaltyObjective(
                self.control, new_weight, f"scaled_{self.name}"
            )
            return scaled_obj
        else:
            raise TypeError(f"Cannot multiply ChangePenaltyObjective with {type(mul)}")

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
        results = pd.Series(weight**2 * diff**2 * ts)
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

    def __add__(self, other):
        """Add another objective to this CombinedObjective"""
        if isinstance(other, CombinedObjective):
            return CombinedObjective(
                *self.objectives, *other.objectives, normalization=self.normalization
            )
        else:
            raise TypeError(f"Cannot add CombinedObjective with {type(other)}")

    def __mul__(self, other):
        """Scale all objectives in the combination"""
        if isinstance(other, (int, float, CasadiParameter)):
            scaled_objectives = [obj * other for obj in self.objectives]
            return CombinedObjective(
                *scaled_objectives, normalization=self.normalization
            )
        else:
            raise TypeError(f"Cannot multiply CombinedObjective with {type(other)}")

    def get_delta_u_objectives(self):
        """Returns a list of all ChangePenaltyObjective instances"""
        return [
            obj for obj in self.objectives if isinstance(obj, ChangePenaltyObjective)
        ]

    def get_casadi_expression(self):
        """Combine all objectives into a single CasADi expression"""
        terms = []
        for obj in self.objectives:
            terms.append(obj.get_weighted_expression())
        return sum(terms) / self.normalization

    def calculate_values(self, result_df, grid):
        """Calculate values for each objective component using the result dataframe"""
        self._values = {}
        df = self._prepare_dataframe(result_df, grid)
        total_value = 0

        # For control change penalties, we also need the previous control to penalise the first step
        if grid is not None:
            current_start_index = result_df.index.get_loc(grid[0])  # Get time step 0
            previous_start_index = current_start_index - 1  # Get previous time step
            # Update grid and dataframe
            if previous_start_index >= 0:
                previous_time_step = result_df.index[previous_start_index]
                helper_grid = np.insert(grid, 0, previous_time_step)
            else:
                helper_grid = grid
            df_helper = self._prepare_dataframe(result_df, helper_grid)
        else:
            df_helper = self._prepare_dataframe(result_df, grid)

        for obj in self.objectives:
            name = obj.name
            if isinstance(obj, ChangePenaltyObjective):
                # Handle symbolic or numeric weights for control change penalties
                if isinstance(obj.weight, CasadiParameter):
                    weight_name = obj.weight.name
                    weight = df_helper.loc[:, ("parameter", weight_name)].shift(-1).iloc[:-1]
                elif isinstance(obj.weight, CompositeWeight):
                    weight = obj.weight.evaluate(df_helper).shift(-1).iloc[:-1]
                else:
                    weight = obj.weight

                control_name = obj.get_control_name()
                control_series = df_helper.loc[:, ("variable", control_name)]
                value = obj.calculate_value(control_series, weight)
                self._values[name] = value / self.normalization
            else:
                if isinstance(obj.weight, (CasadiParameter)):
                    weight_name = obj.weight.name
                    weight = df.loc[:, ("parameter", weight_name)].iloc[:-1]
                elif isinstance(obj.weight, CompositeWeight):
                    weight = obj.weight.evaluate(df).iloc[:-1]

                else:
                    weight = obj.weight

                if isinstance(obj, SubObjective):
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

            active_objectives[self.default_objective] = (
                active_objectives[self.default_objective] & ~condition_mask
            )

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

        if "@" in condition_str:
            condition_str = _replace_subexpressions(condition_str)

        if "?" in condition_str:
            condition_str = _replace_ternary(condition_str)

        # Replace CasADi logical operators with Python equivalents for eval
        condition_str = condition_str.replace("&&", " and ")
        condition_str = condition_str.replace("||", " or ")
        condition_str = re.sub(r"(?<![<>=!])!(?!=)", " not ", condition_str)

        identifier_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*'
        potential_vars = re.findall(identifier_pattern, condition_str)

        operators_keywords = {
            "and",
            "or",
            "not",
            "in",
            "is",
            "if",
            "else",
            "elif",
            "for",
            "while",
            "def",
            "class",
        }
        var_names = [var for var in potential_vars if var not in operators_keywords]

        var_names = list(dict.fromkeys(var_names))

        values_dict = {}

        for var_name in var_names:
            if var_name == "time":
                values_dict["time"] = df.index.to_numpy()
                continue

            for col_type in ["variable", "parameter"]:
                if (col_type, var_name) in df.columns:
                    values_dict[var_name] = df.loc[:, (col_type, var_name)].values
                    break

        n_rows = len(df)
        mask = np.zeros(n_rows, dtype=bool)

        for i in range(n_rows):
            local_vars = {}
            for var_name, values in values_dict.items():
                local_vars[var_name] = values[i]
            try:
                eval_str = condition_str
                result = eval(eval_str, {"__builtins__": {}, "abs": abs, "min": min, "max": max, "where": np.where}, local_vars)
                mask[i] = bool(result)

            except Exception as e:
                print(f"Error evaluating condition at row {i}: {e}")
                mask[i] = False

        return pd.Series(mask, index=df.index)
