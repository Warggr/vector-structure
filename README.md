# vector-structure

Lightweight utilities for working with _structured flat vectors_ and _block matrices_ in NumPy.

# Example usage

Consider a Newton algorithm for solving the KKT conditions, as described [here](https://won-j.github.io/M1399_000200-2021fall/lectures/22-newton/newton_constr.html):

$$\begin{pmatrix}
    \nabla^2 f(x) & \color{gray}{Df(x)^T} & A^T \\
    \color{gray}{-diag(\lambda) Df(x)} & \color{gray}{-diag(f(x))} & \color{gray}{0} \\
    A & \color{gray}{0} & 0
\end{pmatrix} \begin{pmatrix} x^* \\ \color{gray}{\lambda^\*} \\ v^\* \end{pmatrix} = \begin{pmatrix} \\ \dots \\
\end{pmatrix}$$

where the grey components are only required when inequality constraints are present.

```python
from vector_structure import VectorStructure

structure = [("x", n)]
if ineq_constraints:
    structure.append(("lambda", m))
structure.append(("mu", p))

vs = VectorStructure(structure)

M = np.zeros((vs.size, vs.size))
M[vs["x"], vs["x"]] = nabla2(f)(x)
if ineq_constraints:
    M[vs["x"], vs["lambda"]] = Df(x).T
M[vs["x"], vs["mu"]] = A
...

x_lambda_mu = np.linalg.solve(M, r)
x = x_lambda_mu[vs["x"]]
if ineq_constraints:
    lambda = x_lambda_mu[vs["lambda"]]
```
