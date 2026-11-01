# Two-Phase Simplex Linear Programming Solver

A linear programming solver using the **Two-Phase Simplex Algorithm** that can solve various types of LP problems.

## Features

✅ **Solve LP Problems**
- **Minimization** and **Maximization** problems
- **Positive** (≥ 0), **Negative** (≤ 0), and **Free** (unrestricted) variables
- Constraint types: `<=`, `>=`, `=`

✅ **Advanced Algorithm**
- Phase I: Find initial feasible solution
- Phase II: Optimize the solution
- **Dantzig's rule** (largest reduced cost) for fast progress, with an automatic switch to **Bland's Rule** when the method stalls on degenerate pivots, so it can never cycle
- Vectorized NumPy pivots

✅ **Robust**
- Handles negative RHS constraints
- Handles degenerate Phase I endings (artificial variables left in the basis at zero) and redundant equality constraints
- Detects **unbounded** and **infeasible** problems

## How to Use?

### Installation
````bash
pip install numpy
python Lp_solver.py
````

### Example: Simple Optimization Problem

Suppose you want to solve:
````
maximize:    3*x1 + 2*x2
subject to:
             x1 + x2 <= 4
             2*x1 + x2 = 7
             x1, x2 >= 0
````

**Steps to Input:**

1️⃣ **Number of decision variables:** `2`

2️⃣ **Objective coefficients:** `3 2`

3️⃣ **Problem type:** `yes` (Maximization)

4️⃣ **Number of constraints:** `2`

5️⃣ **First constraint:** `1 1 4` (x1 + x2 <= 4)
   - Type: `<=`

6️⃣ **Second constraint:** `2 1 7` (2*x1 + x2 = 7)
   - Type: `=`

7️⃣ **Variable types:**
   - x1: `1` (positive ≥ 0)
   - x2: `1` (positive ≥ 0)

**Output:**
````
Optimal solution:
x1 = 3.0
x2 = 1.0
Optimal objective value = 11.0
````

---

## Input Format Explanation

### 1. Number of Decision Variables
````
Number of decision variables: 2
````

### 2. Objective Function Coefficients
````
Enter 2 objective coefficients: 3 2
````

### 3. Problem Type
````
Maximization problem? (yes/no): yes
````

### 4. Constraints

For each constraint, enter **variable coefficients + RHS (right-hand side)**:

Example for `x1 + 2*x2 <= 5`:
````
Constraint 1 (coefficients + RHS): 1 2 5
Type (<= / >= / =): <=
````

An invalid type is asked again instead of stopping the program.

### 5. Variable Types
- `1` = Positive or zero (`x >= 0`)
- `2` = Negative or zero (`x <= 0`)
- `3` = Free (unrestricted)

---

## More Examples

### Example 2: Free Variables
````
minimize:    x1 + 2*x2
subject to:
             x1 + x2 = 3
             2*x1 - x2 >= 1
             x1 >= 0, x2 free
````

**Input:**
````
Number of decision variables: 2
Enter 2 objective coefficients: 1 2
Maximization problem? (yes/no): no
Number of constraints: 2
Constraint 1 (coefficients + RHS): 1 1 3
Type (<= / >= / =): =
Constraint 2 (coefficients + RHS): 2 -1 1
Type (<= / >= / =): >=
Variable types:
x1: 1
x2: 3
````

**Output:**
````
Error: UNBOUNDED problem.
````
Because `x2 = 3 - x1`, the objective equals `6 - x1`, and `x1` can grow without limit
(with `x2` going negative), so there is no minimum.

### Example 3: Negative RHS Handling
````
minimize:    x1 - x2
subject to:
             x1 - x2 = -2
             x1, x2 >= 0
````

The program automatically normalizes negative constraints:
`x1 - x2 = -2` → `-x1 + x2 = 2`

---

## Using It from Python

````python
from Lp_solver import solve_lp

# maximize 3 x1 + 2 x2  s.t.  x1 + x2 <= 4,  2 x1 + x2 = 7,  x >= 0
x, obj = solve_lp([3, 2], [([1, 1], "<=", 4), ([2, 1], "=", 7)])
# x = [3., 1.], obj = 11.0

# var_types: 1 = >=0, 2 = <=0, 3 = free;  maximize=False for minimization
x, obj = solve_lp([1, 1], [([1, 1], ">=", 2)], var_types=[1, 3], maximize=False)
````

`solve_lp` raises `ValueError` with `UNBOUNDED problem.` or `INFEASIBLE problem.`.

---

## Code Structure

### Main Class: `TwoPhaseSimplex`

#### Main Methods:

| Method | Description |
|--------|-------------|
| `read_problem()` | Read problem input |
| `load_problem()` | Set the problem from Python (no prompts) |
| `build_tableau()` | Build initial tableau and Phase I |
| `run_simplex()` | Execute simplex algorithm |
| `end_phase1()` | Check feasibility, remove artificial variables |
| `build_phase2_objective()` | Set up Phase II objective |
| `optimize()` | Solve the loaded problem, return `(x, objective)` |
| `solve()` | Read, solve and print the complete problem |

#### Helper Functions:
````python
safe_int(prompt)              # Safe integer input
safe_float_list(prompt, n)    # Safe list input
safe_choice(prompt, choices)  # Safe choice input (constraint type)
solve_lp(c, constraints, var_types=None, maximize=True)
````

---

## Technical Details

### Negative RHS Handling
When constraint RHS is negative, multiply entire constraint by -1:
````python
x <= -1  →  -x >= 1
x >= -1  →  -x <= 1
x  = -1  →  -x  = 1
````

### Artificial Variables Left in the Basis After Phase I
Phase I can end with an artificial variable still in the basis at value zero.
Simply deleting it leaves its row without a basic variable, and Phase II can then
violate that constraint (earlier versions returned infeasible "optimal" points this way).
Now the artificial is **pivoted out** on any non-zero entry of its row; if the row has
none, the constraint is redundant and the row is **removed**.

### Pivoting Rule
- Normally **Dantzig's rule**: the column with the largest reduced cost enters, which
  needs far fewer iterations than always taking the first eligible column.
- After `DEGENERATE_LIMIT` (50) consecutive degenerate pivots it switches to
  **Bland's Rule** (smallest index enters, ties in the ratio test broken by smallest basic
  variable index), which guarantees termination, and switches back after progress.

### ⚙️ Algorithm Parameters
````python
EPS = 1e-9               # pivot / reduced-cost tolerance
FEAS_TOL = 1e-7          # feasibility tolerance (relative to the largest |RHS|)
DEGENERATE_LIMIT = 50    # degenerate pivots before switching to Bland's rule
MAX_ITER = 100_000       # safety limit on pivots
````

---

## Performance

Pivots, the ratio test and the Phase I/II set-up are vectorized NumPy operations, and
Dantzig's rule reduces the number of pivots. Average solve time on random feasible LPs
(same problems, same machine, tableau printing disabled in both):

| Size (variables × constraints) | Previous version | Current version | Speed-up |
|---|---|---|---|
| 30 × 30   | 0.024 s | 0.003 s | ~8× |
| 60 × 60   | 0.163 s | 0.018 s | ~9× |
| 100 × 100 | 0.928 s | 0.056 s | ~17× |
| 150 × 150 | 3.683 s | 0.146 s | ~25× |

---

## Error Messages

### ❌ **UNBOUNDED Problem**
````
Error: UNBOUNDED problem.
````
**Meaning:** Objective function can improve infinitely.

### ❌ **INFEASIBLE Problem**
````
Error: INFEASIBLE problem.
````
**Meaning:** No solution satisfies all constraints.

### ❌ **Input Error**
````
Expected N numbers.
Enter numeric values separated by space.
````

---

## Viewing Solution Steps

To see the tableau after every pivot:
````bash
python Lp_solver.py --steps
````

---

## Running the Tests
````bash
pip install pytest scipy
python -m pytest tests
````
SciPy is only used by the tests, as an independent reference solver.

---

## Requirements
````
numpy >= 1.20
python >= 3.9
````

---

## References

- **Two-Phase Simplex Algorithm:** Classical linear programming methods
- **Bland's Rule:** Prevents cycling in simplex
- **Constraint Normalization:** Handling negative RHS

---

## License

Free to use and modify.

---

## Author

👨‍💻 Payamkou
