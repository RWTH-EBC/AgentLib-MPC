import pandas as pd
import numpy as np
import casadi as ca
import re

class EqObjective:
    def __init__(self, expressions, weight: float = 1.0, name: str = None):
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
        self.is_complex_expr = hasattr(expressions, 'dep')

    def get_weighted_expression(self):
        """Returns the final weighted expression"""
        if hasattr(self.weight, 'sym'):
            weight_value = self.weight.sym
        else:
            weight_value = self.weight
        return weight_value * self.expression

    def calculate_value(self, data, weight):
        """Calculate the objective value from data"""
        ts = np.diff(data.index)
        if isinstance(data, pd.Series):
            return sum(weight * data.values[:-1] * ts)
        result = self._evaluate_expression(self.expression, data)
        return sum(weight * result * ts)

    def _evaluate_expression(self, expr, df):
        """Evaluate a complex expression using dataframe values"""
        if hasattr(expr, 'name') and not hasattr(expr, 'dep'):
            try:
                if callable(expr.name):
                    var_name = expr.name()
                else:
                    var_name = expr.name
                for col_type in ['variable', 'parameter']:
                    if (col_type, var_name) in df.columns:
                        return df.loc[:, (col_type, var_name)].values[:-1]
                if var_name in df.columns:
                    return df.loc[:, var_name].values[:-1]
            except Exception as e:
                print(f"Name access error: {e}")

        expr_str = str(expr)

        var_names = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*', expr_str)

        if var_names:
            values_found = {}
            for var_name in var_names:
                for col_type in ['variable', 'parameter']:
                    if (col_type, var_name) in df.columns:
                        values_found[var_name] = df.loc[:, (col_type, var_name)].values[:-1]
                        break

            if len(values_found) == len(var_names):
                if '+' in expr_str:
                    return sum(values_found.values())
                elif '*' in expr_str:
                    result = 1
                    for val in values_found.values():
                        result = result * val
                    return result
                elif '-' in expr_str and len(values_found) == 2:
                    keys = list(values_found.keys())
                    return values_found[keys[0]] - values_found[keys[1]]
                elif '/' in expr_str and len(values_found) == 2:
                    keys = list(values_found.keys())
                    return values_found[keys[0]] / values_found[keys[1]]

            elif len(values_found) > 0:
                return next(iter(values_found.values()))

        if isinstance(expr, (int, float)):
            return np.full(len(df.index) - 1, expr)
        try:
            const_val = float(expr)
            return np.full(len(df.index) - 1, const_val)
        except:
            pass

        raise ValueError(f"Unable to evaluate expression: {expr}")


class SqObjective(EqObjective):
    """Objective term that squares an expression"""

    def __init__(self, expressions, weight=1.0, name=None):
        """
        Create an objective term that squares an expression

        Args:
            expressions: Expression to be squared (can be a complex expression)
            weight: Weight factor for this objective
            name: Optional name for identification
        """
        super().__init__(expressions=expressions, weight=weight, name=name)

    def get_weighted_expression(self):
        """Returns the squared expression with weight"""
        if hasattr(self.weight, 'sym'):
            weight_value = self.weight.sym
        else:
            weight_value = self.weight
        return (weight_value ** 2) * (self.expression ** 2)

    def calculate_value(self, data, weight):
        """Calculate value using the expression"""
        ts = np.diff(data.index)
        if isinstance(data, pd.Series):
            result = data.values[:-1] ** 2
            return sum((weight ** 2) * result * ts)

        result = self._evaluate_expression(self.expression, data)
        return sum((weight ** 2) * (result ** 2) * ts)

    # def get_expression_name(self):
    #     """Returns a representative name for the expression"""
    #     names = self._extract_variable_names(self.expression)
    #     if len(names) == 1:
    #         return names[0]
    #     return f"expr_{'_'.join(names)}"



class DeltaUObjective(EqObjective):
    def __init__(self, expressions, weight: float = 1.0, name: str = None):
        """
        Args:
            expressions: Control variable to track changes
            weight: Weight factor for this objective
            name: Optional name for identification/reporting
        """
        self.control = expressions
        super().__init__(expressions=expressions, weight=weight,
                         name=name or f"delta_{self._get_control_name(expressions)}")

    def _get_control_name(self, control):
        """Get the name of the control variable safely"""
        if hasattr(control, 'name'):
            if callable(control.name):
                try:
                    return control.name()
                except Exception:
                    pass
            else:
                return control.name
        return str(control)

    def get_control_name(self):
        """Return the name of the associated control variable"""
        return self._get_control_name(self.control)

    def get_weighted_expression(self):
        """
        Override parent method to provide a placeholder.
        The actual penalty calculation happens in the discretization step.
        """
        # Include weight in expression to ensure it's properly registered
        if hasattr(self.weight, 'sym'):
            return 0 * self.weight.sym
        return 0

    def calculate_value(self, series, weight):
        """Returns the final weighted result by multiplying all expressions"""
        diff_values = series.diff()
        diff = diff_values.values[1:]  # Skip first NaN value
        ts = np.diff(series.index)
        results = weight ** 2 * diff ** 2 * ts
        if isinstance(results, pd.Series):
            return sum(results.dropna())
        elif isinstance(results, np.ndarray):
            return np.nansum(results)
        else:
            return sum(results)

    def get_weight(self):
        """Return the weight in a way that ensures proper symbolic handling"""
        if hasattr(self.weight, 'sym'):
            return self.weight.sym
        return self.weight


class FullObjective:
    """Container for multiple objective terms with normalization"""

    def __init__(self, *objectives, normalization=1.0):
        """
        Args:
            *objectives: Variable number of objective terms
            normalization: Global normalization factor
        """
        self.objectives = list(objectives)
        self.normalization = normalization
        self._values = {}

    def get_delta_u_objectives(self):
        """Returns a list of all DeltaUObjective instances"""
        return [obj for obj in self.objectives if isinstance(obj, DeltaUObjective)]

    def get_sq_objectives(self):
        """Returns a list of all SqObjective instances"""
        return [obj for obj in self.objectives if isinstance(obj, SqObjective)]

    def get_regular_objectives(self):
        """Returns a list of all regular EqObjective instances"""
        return [obj for obj in self.objectives if not isinstance(obj, (DeltaUObjective, SqObjective))]

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
            if hasattr(obj.weight, 'sym'):
                weight_name = obj.weight.name
                if ('parameter', weight_name) in df.columns:
                    weight = df.loc[:, ('parameter', weight_name)].iloc[:-1]
                else:
                    weight = obj.weight
            else:
                weight = obj.weight
            try:
                if isinstance(obj, DeltaUObjective):
                    control_name = obj.get_control_name()
                    control_series = df.loc[:, ('variable', control_name)]
                    value = obj.calculate_value(control_series, weight)
                    self._values[name] = value / self.normalization
                elif isinstance(obj, SqObjective):
                    value = obj.calculate_value(df, weight)
                    self._values[name] = value / self.normalization
                elif isinstance(obj, EqObjective):
                    value = obj.calculate_value(df, weight)
                    self._values[name] = value / self.normalization
                if self._values[name] is not None:
                    total_value += self._values[name]
            except Exception as e:
                self._values[name] = None
                print(f"Error calculating {name}: {e}")
                import traceback
                traceback.print_exc()

        self._values['total'] = total_value
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
        time_value = None

        # Multiindex handling
        if len(df.index) > 0 and isinstance(df.index[0], str) and '(' in df.index[0]:
            new_indices = []
            for idx in df.index:
                try:
                    parts = idx.strip('()').split(',', 1)
                    new_indices.append(float(parts[1].strip()))
                except (ValueError, IndexError):
                    new_indices.append(idx)
            new_df.index = new_indices

        for col in new_df.columns:
            if col[0] in ['upper', 'lower']:
                continue
            if col[0] == 'parameter':
                new_df[col] = new_df[col].ffill()
            elif col[0] == 'variable':
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


import casadi as ca


class ConditionalObjective:
    """Represents a conditional objective that switches between different objectives based on conditions"""

    def __init__(self, *condition_objective_pairs, default_objective=None):
        """
        Args:
            *condition_objective_pairs: Tuples of (condition, objective)
                where condition is a CasADi expression that evaluates to True/False
                and objective is a FullObjective
            default_objective: The objective to use when all conditions are False
        """
        self.condition_objective_pairs = condition_objective_pairs
        self.default_objective = default_objective or FullObjective()

        # Collect all objectives
        self.all_objectives = [self.default_objective]
        for _, objective in condition_objective_pairs:
            if objective not in self.all_objectives:
                self.all_objectives.append(objective)

        # Flatten all individual objective terms for reporting
        self._flattened_objectives = []
        for obj in self.all_objectives:
            if hasattr(obj, 'objectives'):
                self._flattened_objectives.extend(obj.objectives)

    @property
    def objectives(self):
        """Return flattened list of all objective terms for reporting"""
        return self._flattened_objectives

    def get_casadi_expression(self):
        """Combine all objectives into a conditional CasADi expression"""
        result = self.default_objective.get_casadi_expression()

        # Apply each condition in reverse order
        for condition, objective in reversed(self.condition_objective_pairs):
            result = ca.if_else(condition, objective.get_casadi_expression(), result)

        return result

    def get_delta_u_objectives(self):
        """Returns all DeltaUObjective instances from all contained objectives"""
        all_delta_u = []
        for objective in self.all_objectives:
            all_delta_u.extend(objective.get_delta_u_objectives())
        return list(set(all_delta_u))  # Remove duplicates

    def get_sq_objectives(self):
        """Returns all SqObjective instances"""
        all_sq = []
        for objective in self.all_objectives:
            all_sq.extend(objective.get_sq_objectives())
        return list(set(all_sq))

    def get_regular_objectives(self):
        """Returns all regular EqObjective instances"""
        all_regular = []
        for objective in self.all_objectives:
            all_regular.extend(objective.get_regular_objectives())
        return list(set(all_regular))

    def calculate_values(self, result_df, grid):
        """Calculate values for each objective component."""
        all_values = {}
        total_value = 0

        # Calculate values for all objectives
        for objective in self.all_objectives:
            values = objective.calculate_values(result_df, grid)

            for name, value in values.items():
                if name == 'total':
                    continue  # Skip total as we'll calculate it differently

                all_values[name] = value

                if value is not None:
                    total_value += value

        all_values['total'] = total_value
        return all_values
