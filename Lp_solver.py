import argparse

import numpy as np

EPS = 1e-9               # pivot / reduced-cost tolerance
FEAS_TOL = 1e-7          # feasibility tolerance (relative to the largest |RHS|)
DEGENERATE_LIMIT = 50    # consecutive degenerate pivots before switching to Bland's rule
MAX_ITER = 100_000       # safety net against numerical stalling


def safe_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid integer. Try again.")


def safe_float_list(prompt, expected_len):
    while True:
        try:
            vals = list(map(float, input(prompt).split()))
            if len(vals) != expected_len:
                print(f"Expected {expected_len} numbers.")
                continue
            return vals
        except ValueError:
            print("Enter numeric values separated by space.")


def safe_choice(prompt, choices):
    while True:
        ans = input(prompt).strip()
        if ans in choices:
            return ans
        print(f"Choose one of: {' / '.join(choices)}")


def _clean(v):
    """Round for display and avoid printing -0.0."""
    return round(float(v), 6) + 0.0


class TwoPhaseSimplex:
    def __init__(self):
        self.n = 0
        self.c = None
        self.is_max = True
        self.constraints = []
        self.var_types = []
        self.tableau = None
        self.basis = []
        self.artificial_cols = []
        self.col_map = []
        self.var_matrix = None   # x = var_matrix @ (decision columns of the tableau)
        self.n_dec = 0           # number of decision columns in the tableau
        self.iterations = 0

    # ---------------------------
    # Input
    # ---------------------------
    def load_problem(self, c, constraints, var_types=None, maximize=True):
        """
        Set the problem without prompting.

        c           : objective coefficients
        constraints : list of (coefficients, sense, rhs), sense in "<=", ">=", "="
        var_types   : 1 = >=0, 2 = <=0, 3 = free  (default: all >=0)
        """
        self.n = len(c)
        self.is_max = maximize
        self.c = np.asarray(c, dtype=float) * (1.0 if maximize else -1.0)
        self.constraints = []
        for coeffs, sense, rhs in constraints:
            coeffs = np.asarray(coeffs, dtype=float)
            if coeffs.shape != (self.n,):
                raise ValueError(f"Constraint needs {self.n} coefficients.")
            if sense not in ("<=", ">=", "="):
                raise ValueError("Invalid constraint type.")
            self.constraints.append((coeffs, sense, float(rhs)))
        self.var_types = [1] * self.n if var_types is None else list(var_types)
        if len(self.var_types) != self.n or any(t not in (1, 2, 3) for t in self.var_types):
            raise ValueError("Variable types must be 1, 2 or 3 (one per variable).")

    def read_problem(self):
        self.n = safe_int("Number of decision variables: ")
        self.c = np.array(
            safe_float_list(f"Enter {self.n} objective coefficients: ", self.n)
        )
        ans = input("Maximization problem? (yes/no): ").lower()
        self.is_max = ans in ["yes", "y"]
        if not self.is_max:
            self.c = -self.c

        self.constraints = []
        m = safe_int("Number of constraints: ")
        print("Allowed types: <=, >=, =")
        for i in range(m):
            row = safe_float_list(f"Constraint {i+1} (coefficients + RHS): ", self.n + 1)
            sense = safe_choice("Type (<= / >= / =): ", ["<=", ">=", "="])
            self.constraints.append((np.array(row[:-1]), sense, row[-1]))

        self.var_types = []
        print("Variable types:")
        print("1 = >=0, 2 = <=0, 3 = free")
        for i in range(self.n):
            while True:
                t = safe_int(f"x{i+1}: ")
                if t in [1, 2, 3]:
                    self.var_types.append(t)
                    break
                print("Choose 1, 2, or 3.")

    # ---------------------------
    # Build Tableau
    # ---------------------------
    def build_tableau(self):
        n, m = self.n, len(self.constraints)

        # Map decision variables to non-negative tableau columns:
        #   type 1: x = y      type 2: x = -y      type 3: x = y+ - y-
        self.col_map = []
        col = 0
        for t in self.var_types:
            width = 2 if t == 3 else 1
            self.col_map.append(list(range(col, col + width)))
            col += width
        self.n_dec = col
        M = np.zeros((n, self.n_dec))
        for j, (t, cols) in enumerate(zip(self.var_types, self.col_map)):
            M[j, cols[0]] = -1.0 if t == 2 else 1.0
            if t == 3:
                M[j, cols[1]] = -1.0
        self.var_matrix = M

        A = np.array([coeffs for coeffs, _, _ in self.constraints], dtype=float).reshape(m, n)
        b = np.array([rhs for _, _, rhs in self.constraints], dtype=float)
        senses = [sense for _, sense, _ in self.constraints]

        # Negative RHS: multiply the row by -1 and flip the inequality
        flip = {"<=": ">=", ">=": "<=", "=": "="}
        neg = b < 0
        A[neg] *= -1
        b[neg] *= -1
        senses = [flip[s] if f else s for s, f in zip(senses, neg)]

        slack_count = sum(s != "=" for s in senses)
        artificial_count = sum(s != "<=" for s in senses)
        total_cols = self.n_dec + slack_count + artificial_count

        T = np.zeros((m + 1, total_cols + 1))
        T[:m, :self.n_dec] = A @ M
        T[:m, -1] = b

        self.basis = [0] * m
        self.artificial_cols = []
        self.slack_cols = [None] * m   # slack/surplus column of each row
        slack_ptr = self.n_dec
        art_ptr = self.n_dec + slack_count
        for i, sense in enumerate(senses):
            if sense != "=":
                T[i, slack_ptr] = 1.0 if sense == "<=" else -1.0
                self.slack_cols[i] = slack_ptr
                if sense == "<=":
                    self.basis[i] = slack_ptr
                slack_ptr += 1
            if sense != "<=":
                T[i, art_ptr] = 1.0
                self.basis[i] = art_ptr
                self.artificial_cols.append(art_ptr)
                art_ptr += 1

        # Phase I objective: maximise -(sum of artificials), already in canonical form
        art_rows = [i for i, col in enumerate(self.basis) if col >= self.n_dec + slack_count]
        T[-1] = T[art_rows].sum(axis=0)
        T[-1, self.artificial_cols] = 0.0

        self.tableau = T
        self.rhs_scale = max(1.0, float(np.abs(b).max(initial=0.0)))

    # ---------------------------
    # Pivot
    # ---------------------------
    def pivot(self, r, c):
        T = self.tableau
        T[r] /= T[r, c]
        col = T[:, c].copy()
        col[r] = 0.0
        rows = np.flatnonzero(col)
        if 2 * rows.size > T.shape[0]:
            T -= np.outer(col, T[r])            # dense column: update in place
        else:
            T[rows] -= np.outer(col[rows], T[r])
        T[:, c] = 0.0
        T[r, c] = 1.0
        self.basis[r] = c

    # ---------------------------
    # Simplex Iteration
    # ---------------------------
    def run_simplex(self, print_steps=False):
        """
        Primal simplex on the current tableau.

        Uses Dantzig's rule (largest reduced cost) for speed and falls back to
        Bland's rule after DEGENERATE_LIMIT consecutive degenerate pivots, which
        rules out cycling.
        """
        T = self.tableau
        m = len(self.basis)
        degenerate = 0
        for _ in range(MAX_ITER):
            costs = T[-1, :-1]
            bland = degenerate >= DEGENERATE_LIMIT
            if bland:
                candidates = np.flatnonzero(costs > EPS)
                if candidates.size == 0:
                    return
                entering = int(candidates[0])
            else:
                entering = int(np.argmax(costs))
                if costs[entering] <= EPS:
                    return

            column = T[:m, entering]
            rows = np.flatnonzero(column > EPS)
            if rows.size == 0:
                raise ValueError("UNBOUNDED problem.")
            ratios = np.maximum(T[rows, -1], 0.0) / column[rows]
            best = ratios.min()
            ties = rows[ratios <= best + EPS]
            if bland:
                leaving = int(min(ties, key=lambda i: self.basis[i]))
            else:
                leaving = int(ties[np.argmax(column[ties])])   # most stable pivot

            degenerate = degenerate + 1 if best <= EPS else 0
            self.pivot(leaving, entering)
            self.iterations += 1
            if print_steps:
                print(f"\nIteration {self.iterations}")
                print(np.round(self.tableau, 6))
        raise RuntimeError("Iteration limit reached (numerical stalling).")

    # ---------------------------
    # Phase I -> Phase II
    # ---------------------------
    def end_phase1(self):
        """
        Check feasibility, then remove the artificial variables.

        An artificial that is still basic (at value 0) is pivoted out on any
        non-zero entry of its row; if the row has none it is redundant and is
        deleted. Leaving such a row without a basic variable would let Phase II
        violate the original constraint.
        """
        T = self.tableau
        if T[-1, -1] > FEAS_TOL * self.rhs_scale:
            raise ValueError("INFEASIBLE problem.")

        is_art = np.zeros(T.shape[1] - 1, dtype=bool)
        is_art[self.artificial_cols] = True
        redundant = set()
        for i, col in enumerate(self.basis):
            if not is_art[col]:
                continue
            row = np.abs(T[i, :-1])
            row[is_art] = 0.0
            j = int(np.argmax(row))
            if row[j] > FEAS_TOL:
                T[i, -1] = 0.0       # degenerate pivot keeps every RHS unchanged
                self.pivot(i, j)
            else:
                redundant.add(i)

        keep_rows = [i for i in range(len(self.basis)) if i not in redundant]
        keep_cols = np.flatnonzero(~is_art)
        new_index = np.full(is_art.size, -1)
        new_index[keep_cols] = np.arange(keep_cols.size)
        self.tableau = T[np.ix_(keep_rows + [T.shape[0] - 1],
                                np.append(keep_cols, T.shape[1] - 1))]
        self.basis = [int(new_index[self.basis[i]]) for i in keep_rows]
        self.slack_cols = [None if self.slack_cols[i] is None else int(new_index[self.slack_cols[i]])
                           for i in keep_rows]
        self.artificial_cols = []

    # ---------------------------
    # Phase II Objective
    # ---------------------------
    def build_phase2_objective(self):
        T = self.tableau
        obj = np.zeros(T.shape[1])
        obj[:self.n_dec] = self.c @ self.var_matrix
        # Canonical form: zero the reduced costs of the basic variables
        obj -= obj[self.basis] @ T[:-1]
        T[-1] = obj

    # ---------------------------
    # Solution
    # ---------------------------
    def extract_solution(self):
        T = self.tableau
        values = np.zeros(T.shape[1] - 1)
        values[self.basis] = T[:-1, -1]
        solution = self.var_matrix @ values[:self.n_dec]
        obj_val = float(self.c @ solution)
        if not self.is_max:
            obj_val = -obj_val
        return solution, obj_val

    def optimize(self, print_steps=False):
        """Solve the loaded problem; returns (solution, objective value)."""
        self.iterations = 0
        self.build_tableau()

        if print_steps:
            print("\nPhase I...")
        self.run_simplex(print_steps)
        self.end_phase1()

        if print_steps:
            print("\nPhase II...")
        self.build_phase2_objective()
        self.run_simplex(print_steps)
        return self.extract_solution()

    # ---------------------------
    # Solve LP
    # ---------------------------
    def solve(self, print_steps=False):
        self.read_problem()
        solution, obj_val = self.optimize(print_steps)

        print("\nOptimal solution:")
        for i in range(self.n):
            print(f"x{i+1} = {_clean(solution[i])}")
        print("Optimal objective value =", _clean(obj_val))


def solve_lp(c, constraints, var_types=None, maximize=True):
    """Convenience wrapper: returns (solution, objective value)."""
    solver = TwoPhaseSimplex()
    solver.load_problem(c, constraints, var_types, maximize)
    return solver.optimize()


# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-phase simplex LP solver")
    parser.add_argument("--steps", action="store_true",
                        help="print the tableau after every pivot")
    args = parser.parse_args()
    try:
        solver = TwoPhaseSimplex()
        solver.solve(print_steps=args.steps)
    except (ValueError, RuntimeError) as e:
        print("Error:", e)
    except (EOFError, KeyboardInterrupt):
        print("\nInput aborted.")
