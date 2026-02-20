import numpy as np

EPS = 1e-9


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

    # ---------------------------
    # Input
    # ---------------------------
    def read_problem(self):
        self.n = safe_int("Number of decision variables: ")
        self.c = np.array(
            safe_float_list(f"Enter {self.n} objective coefficients: ", self.n)
        )
        ans = input("Maximization problem? (yes/no): ").lower()
        self.is_max = ans in ["yes", "y"]
        if not self.is_max:
            self.c = -self.c

        m = safe_int("Number of constraints: ")
        print("Allowed types: <=, >=, =")
        for i in range(m):
            row = safe_float_list(f"Constraint {i+1} (coefficients + RHS): ", self.n + 1)
            sense = input("Type (<= / >= / =): ").strip()
            if sense not in ["<=", ">=", "="]:
                raise ValueError("Invalid constraint type.")
            self.constraints.append((np.array(row[:-1]), sense, row[-1]))

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
        self.basis = []
        self.artificial_cols = []
        self.col_map = []

        # Map decision variables
        col = 0
        for t in self.var_types:
            if t in [1, 2]:
                self.col_map.append([col])
                col += 1
            else:  # free variable -> two positive components
                self.col_map.append([col, col + 1])
                col += 2

        base_vars = col
        m = len(self.constraints)

        # -------------------------------------------------------
        # BUG FIX 1: Negative RHS handling
        # -------------------------------------------------------
        normalized = []
        for coeffs, sense, rhs in self.constraints:
            if rhs < -EPS:
                coeffs = -coeffs
                rhs = -rhs
                if sense == "<=":
                    sense = ">="
                elif sense == ">=":
                    sense = "<="
                # "=" stays "="
            normalized.append((coeffs, sense, rhs))

        # Count extra columns (on normalized constraints)
        slack_count = 0
        artificial_count = 0
        for _, sense, _ in normalized:
            if sense == "<=":
                slack_count += 1
            elif sense == ">=":
                slack_count += 1
                artificial_count += 1
            else:
                artificial_count += 1

        total_cols = base_vars + slack_count + artificial_count

        A = np.zeros((m, total_cols))
        b = np.zeros(m)

        slack_ptr = base_vars
        art_ptr = base_vars + slack_count

        for i, (coeffs, sense, rhs) in enumerate(normalized):

            # Decision variables
            for j in range(self.n):
                cols = self.col_map[j]
                if self.var_types[j] == 1:
                    A[i, cols[0]] = coeffs[j]
                elif self.var_types[j] == 2:
                    A[i, cols[0]] = -coeffs[j]
                else:
                    A[i, cols[0]] = coeffs[j]
                    A[i, cols[1]] = -coeffs[j]

            # Slack / Artificial
            if sense == "<=":
                A[i, slack_ptr] = 1
                self.basis.append(slack_ptr)
                slack_ptr += 1
            elif sense == ">=":
                A[i, slack_ptr] = -1
                slack_ptr += 1
                A[i, art_ptr] = 1
                self.basis.append(art_ptr)
                self.artificial_cols.append(art_ptr)
                art_ptr += 1
            else:  # equality
                A[i, art_ptr] = 1
                self.basis.append(art_ptr)
                self.artificial_cols.append(art_ptr)
                art_ptr += 1

            b[i] = rhs

        self.tableau = np.hstack([A, b.reshape(-1, 1)])

        # Phase I objective: minimize sum of artificials -> maximize negative sum
        obj = np.zeros(total_cols)
        for col in self.artificial_cols:
            obj[col] = 1

        phase1_row = np.zeros(total_cols + 1)
        phase1_row[:-1] = -obj
        self.tableau = np.vstack([self.tableau, phase1_row])

        # Canonical form: remove artificial coefficients from Phase I row
        for i, col in enumerate(self.basis):
            if col in self.artificial_cols:
                self.tableau[-1] += self.tableau[i]

    # ---------------------------
    # Pivot
    # ---------------------------
    def pivot(self, r, c):
        self.tableau[r] /= self.tableau[r, c]
        for i in range(self.tableau.shape[0]):
            if i != r:
                self.tableau[i] -= self.tableau[i, c] * self.tableau[r]

    # ---------------------------
    # Simplex Iteration
    # ---------------------------
    def run_simplex(self, print_steps=False):
        m = len(self.constraints)
        iteration = 0
        while True:
            iteration += 1
            costs = self.tableau[-1, :-1]
            # Bland's rule
            entering = next((j for j in range(len(costs)) if costs[j] > EPS), -1)
            if entering == -1:
                return
            ratios = []
            for i in range(m):
                if self.tableau[i, entering] > EPS:
                    ratios.append(self.tableau[i, -1] / self.tableau[i, entering])
                else:
                    ratios.append(np.inf)
            if all(r == np.inf for r in ratios):
                raise ValueError("UNBOUNDED problem.")
            leaving = np.argmin(ratios)
            self.pivot(leaving, entering)
            self.basis[leaving] = entering
            if print_steps:
                print(f"\nIteration {iteration}")
                print(np.round(self.tableau, 6))

    # ---------------------------
    # Phase II Objective
    # ---------------------------
    def build_phase2_objective(self):
        cols = self.tableau.shape[1] - 1
        obj = np.zeros(cols)
        for j in range(self.n):
            mapped = self.col_map[j]
            if self.var_types[j] == 1:
                obj[mapped[0]] = self.c[j]
            elif self.var_types[j] == 2:
                obj[mapped[0]] = -self.c[j]
            else:
                obj[mapped[0]] = self.c[j]
                obj[mapped[1]] = -self.c[j]
        new_row = np.zeros(cols + 1)
        new_row[:-1] = obj
        self.tableau[-1] = new_row

        # -------------------------------------------------------
        # BUG FIX 2: None entries in basis (degenerate Phase I)
        # -------------------------------------------------------
        for i, col in enumerate(self.basis):
            if col is None:
                continue  # artificial degenerate: no contribution to obj row
            coeff = self.tableau[-1, col]
            if abs(coeff) > EPS:
                self.tableau[-1] -= coeff * self.tableau[i]

    # ---------------------------
    # Solve LP
    # ---------------------------
    def solve(self):
        self.read_problem()
        self.build_tableau()

        print("\nPhase I...")
        self.run_simplex(print_steps=True)

        if abs(self.tableau[-1, -1]) > EPS:
            raise ValueError("INFEASIBLE problem.")

        # Remove artificial columns
        keep_cols = [j for j in range(self.tableau.shape[1] - 1)
                     if j not in self.artificial_cols]
        self.tableau = self.tableau[:, keep_cols + [-1]]

        # Remap basis indices to post-removal column positions
        new_basis = []
        for b in self.basis:
            if b in keep_cols:
                new_basis.append(keep_cols.index(b))
            else:
                new_basis.append(None)  # artificial stuck in basis at 0
        self.basis = new_basis

        print("\nPhase II...")
        self.build_phase2_objective()
        self.run_simplex(print_steps=True)

        # Extract solution
        solution = np.zeros(self.n)
        for i, col in enumerate(self.basis):
            if col is not None:
                for j in range(self.n):
                    if col in self.col_map[j]:
                        idx = self.col_map[j].index(col)
                        if self.var_types[j] == 3:
                            if idx == 0:
                                solution[j] += self.tableau[i, -1]
                            else:
                                solution[j] -= self.tableau[i, -1]
                        elif self.var_types[j] == 1:
                            solution[j] = self.tableau[i, -1]
                        else:
                            solution[j] = -self.tableau[i, -1]

        obj_val = self.tableau[-1, -1]
        if self.is_max:
            obj_val = -obj_val

        print("\nOptimal solution:")
        for i in range(self.n):
            print(f"x{i+1} = {round(solution[i], 6)}")
        print("Optimal objective value =", round(obj_val, 6))


# ---------------------------
if __name__ == "__main__":
    try:
        solver = TwoPhaseSimplex()
        solver.solve()
    except Exception as e:
        print("Error:", e)
