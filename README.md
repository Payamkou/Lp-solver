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
- Numerical stability for better precision
- **Bland's Rule** to prevent cycling

✅ **Bug Fixes**
- Handles negative RHS constraints
- Supports degenerate cases (artificial variables at zero)

## How to Use?

### Installation
````bash
python Lp_solver.py
````

### Example: Simple Optimization Problem

Suppose you want to solve:
````
maximize:    3*x1 + 2*x2
subject to:
             x1 + x2 <= 4
             2*x1 + x2 <= 7
             x1, x2 >= 0
````

**Steps to Input:**

1️⃣ **Number of decision variables:** `2`

2️⃣ **Objective coefficients:** `3 2`

3️⃣ **Problem type:** `yes` (Maximization)

4️⃣ **Number of constraints:** `2`

5️⃣ **First constraint:** `1 1 4` (x1 + x2 = 4)
   - Type: `<=`

6️⃣ **Second constraint:** `2 1 7` (2*x1 + x2 = 7)
   - Type: `<=`

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

## Code Structure

### Main Class: `TwoPhaseSimplex`

#### Main Methods:

| Method | Description |
|--------|-------------|
| `read_problem()` | Read problem input |
| `build_tableau()` | Build initial tableau and Phase I |
| `run_simplex()` | Execute simplex algorithm |
| `build_phase2_objective()` | Set up Phase II objective |
| `solve()` | Solve the complete problem |

#### Helper Functions:
````python
safe_int(prompt)           # Safe integer input
safe_float_list(prompt, n) # Safe list input
````

---

## Technical Details

### 🐛 Bug Fixes

#### **BUG FIX 1: Negative RHS Handling**
When constraint RHS is negative, multiply entire constraint by -1:
````python
x <= -1  →  -x >= 1
x >= -1  →  -x <= 1
x  = -1  →  -x  = 1
````

#### **BUG FIX 2: Degenerate Variables in Phase I**
When artificial variable stays in basis at zero value:
````python
if col is None:
    continue  # Skip degenerate artificial variable
````

### ⚙️ Algorithm Parameters
````python
EPS = 1e-9  # Numerical precision for comparisons
````

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

To see all simplex iterations, modify:
````python
self.run_simplex(print_steps=True)  # Display each iteration
````

Output will show intermediate tableaux.

---

## Requirements
````
numpy >= 1.0
python >= 3.6
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
