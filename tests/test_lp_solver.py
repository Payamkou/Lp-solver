import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Lp_solver import TwoPhaseSimplex, solve_lp  # noqa: E402


def test_readme_example_1():
    x, obj = solve_lp([3, 2], [([1, 1], "<=", 4), ([2, 1], "=", 7)])
    assert np.allclose(x, [3, 1])
    assert obj == pytest.approx(11)


def test_readme_example_2_free_variable_is_unbounded():
    # minimize x1 + 2 x2,  x1 + x2 = 3,  2 x1 - x2 >= 1,  x1 >= 0, x2 free
    # x2 = 3 - x1, so the objective 6 - x1 decreases without limit.
    with pytest.raises(ValueError, match="UNBOUNDED"):
        solve_lp([1, 2], [([1, 1], "=", 3), ([2, -1], ">=", 1)],
                 var_types=[1, 3], maximize=False)


def test_free_variable_takes_negative_value():
    # minimize x2 with x2 free, x2 >= -5  -> x2 = -5
    x, obj = solve_lp([0, 1], [([0, 1], ">=", -5), ([1, 0], "<=", 1)],
                      var_types=[1, 3], maximize=False)
    assert x[1] == pytest.approx(-5)
    assert obj == pytest.approx(-5)


def test_readme_example_3_negative_rhs():
    # minimize x1 - x2,  x1 - x2 = -2  ->  every feasible point has objective -2
    x, obj = solve_lp([1, -1], [([1, -1], "=", -2)], maximize=False)
    assert x[0] - x[1] == pytest.approx(-2)
    assert obj == pytest.approx(-2)


def test_artificial_left_in_basis_after_phase1():
    # Phase I ends with an artificial basic at zero in a non-redundant row.
    # The old code dropped it and returned x2 = 0, violating 3 x1 + 2 x2 >= 15.
    x, obj = solve_lp([0, -1], [([1, 2], "<=", 15), ([3, 2], ">=", 15)],
                      var_types=[2, 1])
    assert np.allclose(x, [0, 7.5])
    assert obj == pytest.approx(-7.5)


def test_redundant_equality_rows():
    x, obj = solve_lp([1, 1], [([1, 1], "=", 2), ([2, 2], "=", 4), ([1, 0], "<=", 5)],
                      maximize=False)
    assert x.sum() == pytest.approx(2)
    assert obj == pytest.approx(2)


def test_nonpositive_variable():
    # maximize -x1 with x1 <= 0 and x1 >= -4  ->  x1 = -4
    x, obj = solve_lp([-1], [([1], ">=", -4)], var_types=[2])
    assert x[0] == pytest.approx(-4)
    assert obj == pytest.approx(4)


def test_unbounded():
    with pytest.raises(ValueError, match="UNBOUNDED"):
        solve_lp([1, 1], [([1, -1], "<=", 1)])


def test_infeasible():
    with pytest.raises(ValueError, match="INFEASIBLE"):
        solve_lp([1, 1], [([1, 1], "<=", 1), ([1, 1], ">=", 3)])


def test_degenerate_problem_terminates():
    # Beale's classic cycling example (cycles under the textbook rule without
    # anti-cycling); optimum 0.05 at x = (0.04, 0, 1, 0).
    c = [0.75, -150, 0.02, -6]
    cons = [([0.25, -60, -0.04, 9], "<=", 0),
            ([0.5, -90, -0.02, 3], "<=", 0),
            ([0, 0, 1, 0], "<=", 1)]
    x, obj = solve_lp(c, cons)
    assert obj == pytest.approx(0.05)


def test_interactive_solve(monkeypatch, capsys):
    answers = iter(["2", "3 2", "yes", "2", "1 1 4", "<=", "2 1 7", "=", "1", "1"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    TwoPhaseSimplex().solve()
    out = capsys.readouterr().out
    assert "x1 = 3.0" in out and "x2 = 1.0" in out
    assert "Optimal objective value = 11.0" in out


def test_matches_scipy_on_random_problems():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(0)
    for _ in range(300):
        n, m = rng.integers(2, 8), rng.integers(1, 8)
        c = rng.integers(-5, 10, n).astype(float)
        cons = [(rng.integers(-3, 8, n).astype(float),
                 str(rng.choice(["<=", "<=", ">=", "="])),
                 float(rng.integers(-5, 30))) for _ in range(m)]
        var_types = [int(t) for t in rng.choice([1, 1, 2, 3], n)]
        maximize = bool(rng.integers(0, 2))

        A_ub = [r if s == "<=" else -r for r, s, _ in cons if s != "="]
        b_ub = [b if s == "<=" else -b for _, s, b in cons if s != "="]
        A_eq = [r for r, s, _ in cons if s == "="]
        b_eq = [b for _, s, b in cons if s == "="]
        bounds = [(0, None) if t == 1 else (None, 0) if t == 2 else (None, None)
                  for t in var_types]
        ref = scipy_optimize.linprog(-c if maximize else c,
                                     A_ub=A_ub or None, b_ub=b_ub or None,
                                     A_eq=A_eq or None, b_eq=b_eq or None,
                                     bounds=bounds, method="highs")
        if ref.status == 0:
            x, obj = solve_lp(c, cons, var_types, maximize)
            assert obj == pytest.approx(-ref.fun if maximize else ref.fun, abs=1e-6)
        elif ref.status in (2, 3):
            with pytest.raises(ValueError):
                solve_lp(c, cons, var_types, maximize)
