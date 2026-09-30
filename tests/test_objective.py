"""Fast, solver-free tests for the objective evaluation/reconstruction logic in
agentlib_mpc.data_structures.objective.

Unlike tests/test_examples.py, these tests do not spin up agents or run an
IPOPT solve. They build the example models directly and feed them small,
hand-built result DataFrames, so the CasADi-string reconstruction used for
objective logging/reporting can be exercised in milliseconds. The example
models are reused as-is (not duplicated here) since they already demonstrate
both the current and the deprecated ("legacy") objective notations.
"""

import sys
import unittest
import warnings
from pathlib import Path

import casadi as ca
import numpy as np
import pandas as pd

# "examples" is not an installed package, only importable when the repo root
# is on sys.path. pytest adds it automatically; running this file directly
# (python tests/test_objective.py, or an editor's "Run" button) does not, so
# add it here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentlib_mpc.data_structures.objective import (
    CombinedObjective,
    ConditionalObjective,
    SubObjective,
    _replace_logical_ops,
    _replace_subexpressions,
    _replace_sq_calls,
)
from examples.one_room_mpc.physical.simple_mpc_objective_testing import (
    MyCasadiModel as ConditionalObjectiveModel,
)
from examples.one_room_mpc.physical.simple_mpc_with_time_variant_inputs import (
    MyCasadiModel as LegacyNotationModel,
)


def _make_result_df(index, columns):
    """Build a results-style DataFrame with (col_type, name) MultiIndex columns,
    matching what CombinedObjective/ConditionalObjective.calculate_values expects."""
    col_index = pd.MultiIndex.from_tuples(columns.keys())
    df = pd.DataFrame(index=index, columns=col_index, dtype=float)
    for key, values in columns.items():
        df[key] = values
    return df


class TestReplaceSubexpressions(unittest.TestCase):
    """Unit tests for the @N=... subexpression inlining helper."""

    def test_passthrough_without_at_symbol(self):
        self.assertEqual(_replace_subexpressions("mDot+T_slack"), "mDot+T_slack")

    def test_definition_with_comma_containing_function_is_not_truncated(self):
        # Regression test: fmin(x, y) takes two comma-separated arguments, so a
        # naive "stop at the first comma" parser used to truncate the @1
        # definition to "fmin(x" and mangle the rest of the expression.
        expr_str = "@1=fmin(x,y), (atan2(@1,z)+(2.*@1))"
        self.assertEqual(
            _replace_subexpressions(expr_str),
            "(atan2((fmin(x,y)),z)+(2.*(fmin(x,y))))",
        )

    def test_nested_subexpression_references_are_resolved(self):
        # @2 references @1; both must be fully inlined into the final expression.
        expr_str = "@1=where(a,b,c), @2=@1+d, @2*2"
        self.assertEqual(
            _replace_subexpressions(expr_str),
            "((where(a,b,c))+d)*2",
        )


class TestReplaceSqCalls(unittest.TestCase):
    """Unit tests for the sq(x) -> (x)**2 rewrite helper."""

    def test_sole_sq_call(self):
        self.assertEqual(_replace_sq_calls("sq(x)"), "(x)**2")

    def test_sq_call_within_larger_expression_only_squares_its_own_argument(self):
        # Regression test: squaring used to be applied to the *entire* result
        # whenever "sq(" appeared anywhere in the string, so r*x + s*sq(y)
        # ended up computing (r*x + s*y)**2 instead of r*x + s*y**2.
        expr_str = "(r_mDot*mDot)+(s_T*sq(T_slack))"
        self.assertEqual(
            _replace_sq_calls(expr_str),
            "(r_mDot*mDot)+(s_T*(T_slack)**2)",
        )

    def test_multiple_sq_calls_are_each_squared_independently(self):
        expr_str = "sq(x)+sq(y)"
        self.assertEqual(_replace_sq_calls(expr_str), "(x)**2+(y)**2")


class TestObjectiveEvaluation(unittest.TestCase):
    """Model-level tests built directly from the example models (no agents,
    no solver), so both objective notations get exercised end-to-end."""

    def test_new_notation_conditional_objective_with_comma_regression(self):
        """Covers simple_mpc_objective_testing.py: nested ConditionalObjectives,
        CompositeWeights, and the comma_in_subexpression_regression term added
        for the fmin() comma-parsing bug, all evaluated in one pass."""
        model = ConditionalObjectiveModel()
        time = np.array([0, 100, 300, 500, 650, 900, 1200])
        df = _make_result_df(
            time,
            {
                ("variable", "mDot"): [0.02, 0.0, 0.03, 0.005, 0.02, 0.02, 0.02],
                ("variable", "T"): [298, 292, 295, 290, 294, 296, 293],
                ("variable", "T_slack"): [0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.1],
                ("parameter", "r_mDot"): 1.0,
                ("parameter", "r_mDot2"): 5.0,
                ("parameter", "s_T"): 3.0,
                ("parameter", "switch"): 600.0,
            },
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            values = model.objective.calculate_values(df, None)

        self.assertIn("comma_in_subexpression_regression", values)
        self.assertTrue(np.isfinite(values["total"]))

    def test_old_notation_with_multiple_sq_terms(self):
        """Covers simple_mpc_with_time_variant_inputs.py: the deprecated
        `sum([...])` objective style, wrapped into a single legacy SubObjective
        containing two separate sq(...) terms (s_T*T_slack**2 and
        q_T*(T-T_set)**2). Asserts the reported value matches a hand-computed
        expectation, which the sq()-squares-the-whole-sum bug would break."""
        model = LegacyNotationModel()
        self.assertTrue(getattr(model.objective, "_is_legacy_wrapped", False))

        time = np.array([0, 100, 300])
        df = _make_result_df(
            time,
            {
                ("variable", "mDot"): [0.02, 0.01, 0.03],
                ("variable", "T"): [298, 295, 293],
                ("variable", "T_slack"): [0.0, 0.5, 0.2],
                ("variable", "T_set"): [295, 295, 295],
                ("parameter", "r_mDot"): 1.0,
                ("parameter", "s_T"): 3.0,
                ("parameter", "q_T"): 2.0,
            },
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            values = model.objective.calculate_values(df, None)

        # row0: 1*0.02 + 3*0**2   + 2*(298-295)**2 = 18.02, ts=100
        # row1: 1*0.01 + 3*0.5**2 + 2*(295-295)**2 =  0.76, ts=200
        expected_total = 18.02 * 100 + 0.76 * 200
        self.assertAlmostEqual(values["total"], expected_total)


class TestBareMxAutoWrapping(unittest.TestCase):
    """Regression tests for mixing raw casadi.MX expressions with wrapper
    objects (SubObjective/CombinedObjective) inside CombinedObjective and
    ConditionalObjective.

    CasadiModel.__init__ already auto-wraps a single bare MX at the root of
    the objective tree (backwards-compat shim for the deprecated notation).
    This used to be the *only* place that happened, so mixing a bare MX
    branch (e.g. from plain `sum([...]) / normalization` arithmetic) with a
    wrapper branch built via create_combined_objective(...)/
    create_conditional_objective(...) raised AttributeError deep inside
    get_casadi_expression(). These tests cover the same auto-wrap now
    happening at every level of the tree.
    """

    def test_combined_objective_mixes_bare_mx_and_sub_objective(self):
        x = ca.MX.sym("x")
        y = ca.MX.sym("y")
        wrapped = SubObjective(expressions=y, weight=2, name="wrapped")

        combined = CombinedObjective(x, wrapped, normalization=1.0)

        self.assertIsInstance(combined.objectives[0], SubObjective)
        self.assertIs(combined.objectives[1], wrapped)
        self.assertIsInstance(combined.get_casadi_expression(), ca.MX)

    def test_conditional_objective_mixes_bare_mx_and_combined_objective(self):
        x = ca.MX.sym("x")
        y = ca.MX.sym("y")
        condition = ca.MX.sym("cond")

        combined_branch = CombinedObjective(SubObjective(expressions=y))

        conditional = ConditionalObjective(
            (condition, x), default_objective=combined_branch
        )

        self.assertIsInstance(conditional.get_casadi_expression(), ca.MX)

    def test_pure_wrapper_usage_is_unaffected(self):
        x = ca.MX.sym("x")
        sub = SubObjective(expressions=x, weight=1, name="sub")

        combined = CombinedObjective(sub, normalization=1.0)

        # Already a wrapper object -- must be stored unchanged, not re-wrapped.
        self.assertIs(combined.objectives[0], sub)
        self.assertIsInstance(combined.get_casadi_expression(), ca.MX)

    def test_pure_bare_mx_default_objective_still_works(self):
        # Mirrors the pre-existing root-level wrap in CasadiModel.__init__: a
        # bare MX given as the sole (default) objective should still work.
        x = ca.MX.sym("x")

        conditional = ConditionalObjective(default_objective=x)

        self.assertIsInstance(conditional.default_objective, CombinedObjective)
        self.assertIsInstance(conditional.get_casadi_expression(), ca.MX)


class TestReplaceLogicalOps(unittest.TestCase):
    """Unit tests for the &&/||/! -> logical_and/or/not rewrite helper."""

    def test_and(self):
        self.assertEqual(
            _replace_logical_ops("((1<x)&&(y<2))"),
            "((logical_and((1<x), (y<2))))",
        )

    def test_nested_or_and(self):
        self.assertEqual(
            _replace_logical_ops("((1<x)||((y<2)&&u))"),
            "((logical_or((1<x), logical_and((y<2), u))))",
        )

    def test_not_leaves_not_equal_untouched(self):
        self.assertEqual(
            _replace_logical_ops("((!u)&&(x!=1))"),
            "((logical_and(logical_not(u), (x!=1))))",
        )


class TestConditionEvaluation(unittest.TestCase):
    """Regression tests for the vectorized ConditionalObjective condition
    evaluation. Mapping &&/||/! to NumPy's bitwise &/|/~ only works on boolean
    arrays: a float operand (e.g. a binary input stored as 0.0/1.0) raised a
    TypeError, which was swallowed and turned the whole mask False, and int
    operands were combined bitwise (2 & True == 0)."""

    def setUp(self):
        self.x, self.y, self.u, self.k = (ca.MX.sym(n) for n in "xyuk")
        self.df = _make_result_df(
            np.array([0, 100, 200, 300]),
            {
                ("variable", "x"): [0.5, 1.5, 1.5, 0.5],
                ("variable", "y"): [1.0, 3.0, 1.0, 1.0],
                ("variable", "u"): [1.0, 1.0, 0.0, 0.0],
                ("variable", "k"): [2, 0, 2, 0],
            },
        )

    def _mask(self, condition):
        mask = ConditionalObjective()._evaluate_condition(condition, self.df)
        return mask.tolist()

    def test_and_of_comparisons(self):
        condition = ca.logic_and(self.x > 1, self.y < 2)
        self.assertEqual(self._mask(condition), [False, False, True, False])

    def test_and_with_float_operand(self):
        condition = ca.logic_and(self.u, self.x > 1)
        self.assertEqual(self._mask(condition), [False, True, False, False])

    def test_not_of_float_operand(self):
        condition = ca.logic_not(self.u)
        self.assertEqual(self._mask(condition), [False, False, True, True])

    def test_nested_or_and_with_float_operand(self):
        condition = ca.logic_or(self.x > 1, ca.logic_and(self.y < 2, self.u))
        self.assertEqual(self._mask(condition), [True, True, True, False])

    def test_nonzero_operand_other_than_one_is_true(self):
        # 2 is logically True, even though 2 & True == 0 bitwise.
        condition = ca.logic_and(self.k, self.u > 0.5)
        self.assertEqual(self._mask(condition), [True, False, False, False])

    def test_not_inside_if_else_subexpression(self):
        # CasADi prints this with an @1 subexpression, a ternary and a (!@1).
        condition = ca.if_else(self.x > 1, self.y, self.u) > 0.5
        self.assertEqual(self._mask(condition), [True, True, True, False])


class TestSubObjectiveLogicalOps(unittest.TestCase):
    """Regression tests for &&/||/! inside SubObjective expressions. &&/||
    used to be left untranslated, so the term hit a SyntaxError and was
    logged as 0, and ! became ~, which raised a TypeError on float data."""

    def setUp(self):
        self.x, self.y, self.u = (ca.MX.sym(n) for n in "xyu")
        # Last row is dropped by the evaluation; ts is 100 for every step.
        self.df = _make_result_df(
            np.array([0, 100, 200, 300]),
            {
                ("variable", "x"): [0.5, 1.5, 1.5, 0.5],
                ("variable", "y"): [1.0, 3.0, 1.0, 1.0],
                ("variable", "u"): [1.0, 1.0, 0.0, 0.0],
            },
        )

    def _value(self, expression, name):
        objective = SubObjective(expressions=expression, name=name)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return objective.calculate_value(self.df, 1)

    def test_and_of_comparisons(self):
        expression = ca.if_else(ca.logic_and(self.x > 1, self.y < 2), self.x, 0)
        self.assertAlmostEqual(self._value(expression, "and_cmp"), 1.5 * 100)

    def test_and_with_float_operand(self):
        expression = ca.if_else(ca.logic_and(self.u, self.x > 1), self.y, 0)
        self.assertAlmostEqual(self._value(expression, "and_float"), 3.0 * 100)

    def test_or_with_float_operand(self):
        expression = ca.if_else(ca.logic_or(self.u, self.x > 1), self.y, 0)
        self.assertAlmostEqual(self._value(expression, "or_float"), 5.0 * 100)

    def test_not_of_float_operand(self):
        expression = ca.if_else(ca.logic_not(self.u), self.y, 0)
        self.assertAlmostEqual(self._value(expression, "not_float"), 1.0 * 100)


if __name__ == "__main__":
    unittest.main()
