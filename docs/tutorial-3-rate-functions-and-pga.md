# Tutorial 3 — Rate functions and PGA on multi-terminal objectives

Tutorials 1 and 2 *evaluated* conditional mutual information on a fixed
2-user MAC. This tutorial *optimises* it: we make the per-user precoders
`F_1, F_2` trainable, build a *rate function* from the three pentagon
CMIs, and maximise their sum by projected gradient ascent (PGA) under a
shared total-power budget.

By the end you will understand:

- The general rate-function pattern `f_T = sum_n α_{T,n} I(V_A; V_B | V_C)`
  and how to express it with `Summand` and `evaluate_rate_functions`.
- The closure pattern for PGA: hold tensors as leaves with
  `requires_grad_(True)`, recompute the K-recursion + rate functions
  inside the closure, and return a real scalar.
- How to attach the shared-budget projector `project_total_power` and
  run `pga_ascent` (from the parent library) on the 2-user MAC.

The reference problem is the 2-user MIMO MAC of Tutorials 1 and 2,
optimised so that the achievable rate-region pentagon expands as far as
the shared budget allows. The polished end-to-end version of this
experiment lives in
[`examples/rate_region_maximization.py`](../examples/rate_region_maximization.py).

---

## 1. From one MI to a list of MIs

The MAC rate region is the intersection of three log-det inequalities,
one per non-empty subset of users:

```
   R_1                <=  I_1   = I(X_1; Y | X_2)
            R_2       <=  I_2   = I(X_2; Y | X_1)
   R_1  +   R_2       <=  I_12  = I(X_1, X_2; Y).
```

More generally, every multi-terminal rate region we cover has the form

```
   for every T in S:    sum_{k in T} R_k <= f_T(η, H),
```

where each *rate function* `f_T` is a real-linear combination of
conditional mutual informations:

```
   f_T(η, H) = sum_{n=1}^{N_T} α_{T,n} · I(V_{A_n}; V_{B_n} | V_{C_n}).
```

The `gaussian-dag-cmi` library represents one such summand as a
`Summand` tuple `(α, A, B, C)` and a rate function as a list of summands.
`evaluate_rate_functions` evaluates a whole family of them from a single
K-recursion forward pass.

---

## 2. Build the MAC pentagon as three rate functions

For the MAC pentagon every rate function has just one summand
(`N_T = 1`, `α = 1`); the only thing that changes across the three
inequalities is the triple `(A, B, C)`:

```python
import torch
from gaussian_dag_cmi import (
    compute_k_blocks_multiroot,
    evaluate_rate_functions,
)

torch.manual_seed(0)
d = 4
DTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H_1 = torch.randn(d, d, dtype=DTYPE, device=DEVICE)
H_2 = torch.randn(d, d, dtype=DTYPE, device=DEVICE)
Sigma_root = torch.eye(d, dtype=DTYPE, device=DEVICE)
Sigma_Z    = torch.eye(d, dtype=DTYPE, device=DEVICE)

# The MAC pentagon as three rate functions, each with one summand.
pentagon_inequalities = [
    [(1.0, [0],    [2], [1])],                # I_1  = I(X_1; Y | X_2)
    [(1.0, [1],    [2], [0])],                # I_2  = I(X_2; Y | X_1)
    [(1.0, [0, 1], [2], [])],                 # I_12 = I(X_1, X_2; Y)
]

# Quick check at uniform precoders F_k = I.
F_1 = torch.eye(d, dtype=DTYPE, device=DEVICE)
F_2 = torch.eye(d, dtype=DTYPE, device=DEVICE)

K = compute_k_blocks_multiroot(
    num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
    edge_mats={(2, 0): H_1 @ F_1, (2, 1): H_2 @ F_2},
    root_covs={0: Sigma_root, 1: Sigma_root},
    noise_covs={2: Sigma_Z},
)
I_1, I_2, I_12 = evaluate_rate_functions(K, pentagon_inequalities)
print(f"I_1 = {I_1.item():.4f}, I_2 = {I_2.item():.4f}, I_12 = {I_12.item():.4f}")
```

`evaluate_rate_functions` reuses the same K-recursion output across all
three CMI evaluations, so the cost is one forward pass and three
log-det differences — not three independent recursions.

> **Why `Summand`s with a coefficient and three subsets?** Han–Kobayashi
> inner-bound facets, decode-/compress-and-forward relay bounds, and
> secrecy / leakage trade-offs are all linear combinations of conditional
> MIs with mixed signs. We see one of them — secrecy rate — in
> Tutorial 4.

---

## 3. Make `F_1, F_2` trainable and write the closure

PGA needs a callable that, given the current state of the parameters,
rebuilds the autograd graph and returns the scalar objective:

```python
from gaussian_dag.optimize import pga_ascent
from gaussian_dag.projections import project_total_power

torch.manual_seed(0)
d = 4
P = 8.0                                   # shared total-power budget
DTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H_1 = torch.randn(d, d, dtype=DTYPE, device=DEVICE)
H_2 = torch.randn(d, d, dtype=DTYPE, device=DEVICE)
Sigma_root = torch.eye(d, dtype=DTYPE, device=DEVICE)
Sigma_Z    = torch.eye(d, dtype=DTYPE, device=DEVICE)

# Uniform equal-split initialisation: each ||F_k||_F^2 = P / 2.
scale = (P / (2.0 * d)) ** 0.5
F_1 = (scale * torch.eye(d, dtype=DTYPE, device=DEVICE)).clone().requires_grad_(True)
F_2 = (scale * torch.eye(d, dtype=DTYPE, device=DEVICE)).clone().requires_grad_(True)

pentagon_inequalities = [
    [(1.0, [0],    [2], [1])],
    [(1.0, [1],    [2], [0])],
    [(1.0, [0, 1], [2], [])],
]

def compute_U():
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): H_1 @ F_1, (2, 1): H_2 @ F_2},
        root_covs={0: Sigma_root, 1: Sigma_root},
        noise_covs={2: Sigma_Z},
    )
    I_1, I_2, I_12 = evaluate_rate_functions(K, pentagon_inequalities)
    return I_1 + I_2 + I_12
```

Notice the closure captures `H_1, H_2, Sigma_root, Sigma_Z` from the
enclosing scope, but the only autograd leaves are `F_1` and `F_2`.
Every time `pga_ascent` calls `compute_U()`, a *fresh* graph is
constructed from the current precoder values — exactly what we want.

For a real-valued scalar `U` and complex leaves `F_k`, PyTorch's
`.grad` contains `2 · ∂U / ∂F_k^*` (twice the Wirtinger conjugate-side
derivative). This is the real-Euclidean steepest-ascent direction on
the real and imaginary parts of `F_k`; the factor of 2 is absorbed
into the step size.

---

## 4. The shared-budget projector

For the MAC pentagon, the two precoders share a single total power
budget: `||F_1||_F^2 + ||F_2||_F^2 <= P`. We use the parent library's
`project_total_power`, which rescales every parameter by a single
common factor (the Euclidean projection of the stacked vector onto a
ball of radius `sqrt(P)`):

```python
def projector(params):
    return project_total_power(params, P)
```

`pga_ascent` accepts the projector in two equivalent forms; the
*functional* form above (returning a list of new tensors) avoids the
silent footgun of forgetting an explicit `.copy_` wrapper.

---

## 5. Run PGA

```python
history = pga_ascent(
    compute_U, [F_1, F_2],
    step_size=0.01, num_iters=120, projector=projector,
)
print(f"U: {history[0]:.4f} -> {history[-1]:.4f} nats")
total_power = float(F_1.detach().norm() ** 2 + F_2.detach().norm() ** 2)
print(f"||F_1||_F^2 + ||F_2||_F^2 = {total_power:.4f}  (budget {P:.1f})")
```

`pga_ascent` returns one objective value per iteration, recorded at the
*pre-update* parameter state of that iteration. The final budget should
saturate at `P = 8.0` (the projector pushes every iterate back onto the
boundary of the budget ball whenever the gradient steps out).

The optimised pentagon expands monotonically over the iterations; in
the next section we make that geometry visible.

---

## 6. Recording the pentagon as it expands

To plot the rate region as it evolves, we record `(I_1, I_2, I_12)` on
every iteration. The cleanest way is to record them inside the closure
itself — PGA calls `compute_U()` exactly once per iteration on the
pre-update parameters:

```python
I_1_hist, I_2_hist, I_12_hist = [], [], []

def compute_U_record():
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): H_1 @ F_1, (2, 1): H_2 @ F_2},
        root_covs={0: Sigma_root, 1: Sigma_root},
        noise_covs={2: Sigma_Z},
    )
    I_1, I_2, I_12 = evaluate_rate_functions(K, pentagon_inequalities)
    I_1_hist.append(I_1.item())
    I_2_hist.append(I_2.item())
    I_12_hist.append(I_12.item())
    return I_1 + I_2 + I_12

history = pga_ascent(
    compute_U_record, [F_1, F_2],
    step_size=0.01, num_iters=120, projector=projector,
)
print(f"iter 0:   pentagon (I_1, I_2, I_12) = "
      f"({I_1_hist[0]:.3f}, {I_2_hist[0]:.3f}, {I_12_hist[0]:.3f})")
print(f"iter 119: pentagon (I_1, I_2, I_12) = "
      f"({I_1_hist[-1]:.3f}, {I_2_hist[-1]:.3f}, {I_12_hist[-1]:.3f})")
```

The pentagon expansion across iterations is the qualitative point of
Figure (a) in [`examples/rate_region_maximization.py`](../examples/rate_region_maximization.py)
— the polished end-to-end version of this tutorial, which adds a
`matplotlib` rendering of the nested pentagons and the objective
trajectory.

---

## 7. What is next?

- **Tutorial 4** drops the assumption that the rate-function
  coefficients are non-negative: introducing a sign-indefinite objective
  (the wiretap secrecy rate `I(X;Y) − I(X;Z)`) and the descent companion
  `pga_descent`.
- **Tutorial 5** is the capstone: the same rate-function-sum objective
  scaled to a 12-node random multi-hop network with 9 controllable
  relay processing matrices.
