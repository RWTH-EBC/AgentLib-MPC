import warnings

class SubObjective:
    def __init__(self, expressions, weight: float = 1.0, name: str = None):
        self.expressions = expressions if isinstance(expressions, list) else [expressions]
        self.weight = weight
        self.name = name or f"obj_{id(self)}"
        self._value = None
        self.expr_types = self.identify_types()
        self._validate_expression_types()

    def identify_types(self):
        """
        Identifies the types of all expressions in the objective.
        Returns a list of expression types.
        """
        expr_types = []
        for expression in self.expressions:
            if hasattr(expression, '__class__') and hasattr(expression.__class__, '__name__'):
                expr_types.append(expression.__class__.__name__)
            else:
                expr_types.append(type(expression).__name__)
        return expr_types

    def _validate_expression_types(self):
        """
        Validates that state variables and input variables are not
        mixed in the same objective, as this can lead to optimization issues.
        """
        state_types = ['CasadiState', 'CasadiDifferential', 'State']
        input_types = ['CasadiInput', 'Input']

        has_state = any(expr_type in state_types for expr_type in self.expr_types)
        has_input = any(expr_type in input_types for expr_type in self.expr_types)

        if has_state and has_input:
            state_exprs = [e for e in self.expressions if type(e).__name__ in state_types]
            input_exprs = [e for e in self.expressions if type(e).__name__ in input_types]

            state_names = [e.name if hasattr(e, 'name') else str(e) for e in state_exprs]
            input_names = [e.name if hasattr(e, 'name') else str(e) for e in input_exprs]

            warnings.warn(
                f"SubObjective '{self.name}' mixes state variables {state_names} "
                f"with input variables {input_names}. Resulting in inaccurate calculations of the objective function values"
            )

    def get_weighted_expression(self):
        """Returns the final weighted expression by multiplying all expressions"""
        result = 1
        for expr in self.expressions:
            result = result * expr
        return self.weight * result

    def get_expression_names(self):
        """Returns list of names for all expressions"""
        names = []
        for expr in self.expressions:
            if hasattr(expr, 'name'):
                names.append(expr.name)
            elif hasattr(expr, '_name'):
                names.append(expr._name)
            else:
                # Try to extract name from string representation
                expr_str = str(expr)
                if '(' in expr_str and ')' in expr_str:
                    potential_name = expr_str.split('(')[1].split(')')[0]
                    names.append(potential_name)
                else:
                    names.append(expr_str)
        return names

    def set_value(self, value):
        """Store the calculated value of this objective"""
        self._value = value

    def get_value(self):
        """Get the current value of this objective"""
        return self._value


class SqObjective(SubObjective):
    """Objective term that squares an expression"""

    def __init__(self, expression, weight=1.0, name=None):
        """
        Create an objective term that squares an expression

        Args:
            expression: Expression to be squared
            weight: Weight factor for this objective
            name: Optional name for identification
        """
        self.expression = expression
        super().__init__(expressions=[expression], weight=weight, name=name)

    def get_weighted_expression(self):
        """Returns the squared expression with weight"""
        return self.weight * (self.expression ** 2)

    def get_expression_name(self):
        """Returns the name of the main expression"""
        if hasattr(self.expression, 'name'):
            return self.expression.name
        elif hasattr(self.expression, '_name'):
            return self.expression._name
        else:
            expr_str = str(self.expression)
            if '(' in expr_str and ')' in expr_str:
                return expr_str.split('(')[1].split(')')[0]
            return expr_str


class DeltaUObjective(SubObjective):
    """Objective term for penalizing control changes with optional scaling"""

    def __init__(self, control, weight: float = 1.0, name: str = None, scaling: bool = True):
        """
        Args:
            control: Control variable to track changes
            weight: Weight factor for this objective
            name: Optional name for identification/reporting
            scaling: Whether to apply scaling by average magnitude
        """
        self.control = control
        self.scaling = scaling
        super().__init__(expression=None, weight=weight, name=name or f"delta_{control.name}")

    def get_control_name(self):
        """Return the name of the associated control variable"""
        return self.control.name

    def __str__(self):
        scaling_str = "with scaling" if self.scaling else "without scaling"
        return f"DeltaUObjective({self.name}, control={self.get_control_name()}, weight={self.weight}, {scaling_str})"


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

    # def add_objective(self, objective):
    #     """Add an objective to the list"""
    #     self.objectives.append(objective)
    #     return self

    def get_delta_u_objectives(self):
        """Returns a list of all DeltaUObjective instances"""
        return [obj for obj in self.objectives if isinstance(obj, DeltaUObjective)]

    def get_sq_objectives(self):
        """Returns a list of all SqObjective instances"""
        return [obj for obj in self.objectives if isinstance(obj, SqObjective)]

    def get_regular_objectives(self):
        """Returns a list of all regular SubObjective instances"""
        return [obj for obj in self.objectives if not isinstance(obj, (DeltaUObjective, SqObjective))]

    def get_casadi_expression(self):
        """Combine all objectives into a single CasADi expression"""
        terms = []
        for obj in self.objectives:
            terms.append(obj.get_weighted_expression())
        if terms:
            return sum(terms) / self.normalization
        return 0

    def calculate_values(self):
        """Calculate values for each objective component"""
        self._values = {}
        for obj in self.objectives:
            if hasattr(obj, '_value') and obj._value is not None:
                self._values[obj.name] = obj._value
        return self._values

    def get_values(self):
        """Get all stored objective values"""
        return self._values