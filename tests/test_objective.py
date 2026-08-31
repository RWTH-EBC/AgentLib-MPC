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

import numpy as np
import pandas as pd

# "examples" is not an installed package, only importable when the repo root
# is on sys.path. pytest adds it automatically; running this file directly
# (python tests/test_objective.py, or an editor's "Run" button) does not, so
# add it here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentlib_mpc.data_structures.objective import (
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


if __name__ == "__main__":
    unittest.main()
