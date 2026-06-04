# Tutorial 1 — Installation and your first conditional MI

This first tutorial walks through installing `gaussian-dag-cmi` and
computing the three pentagon conditional mutual informations of a 2-user
vector Gaussian multiple-access channel (MAC). The 2-user MAC is the
smallest non-trivial multi-terminal channel: it has two independent
transmitter inputs and one shared receiver, and its rate region is the
classical pentagon.

By the end of this tutorial you will:

- Have a working `gaussian-dag-cmi` environment.
- Understand the 2-user MAC as a 3-node multi-root linear Gaussian DAG.
- Have evaluated the three pentagon conditional MIs
  `I_1 = I(X_1; Y | X_2)`, `I_2 = I(X_2; Y | X_1)`, and
  `I_{12} = I(X_1, X_2; Y)` with one call each to
  `conditional_mutual_information_from_k`.
- Have checked that the scalar (`d = 1`) values match the classical
  `log(1 + SNR)` formula.

---

## 1. Install the library

`gaussian-dag-cmi` is a small Python package built on PyTorch. It depends
on the parent library [`gaussian-dag`](https://github.com/wadayama/gaussian-dag);
both are installed automatically by `uv sync`. Use
[`uv`](https://docs.astral.sh/uv/) to manage the virtual environment.

```bash
# Clone the repository.
git clone https://github.com/wadayama/gaussian-dag-cmi.git
cd gaussian-dag-cmi

# Install dependencies into a fresh .venv (Python >= 3.12 required).
uv sync
```

Confirm the install:

```bash
uv run pytest
```

You should see all tests pass (one device-parameterised test is skipped
when no CUDA device is available — that is expected on a CPU-only host).

---

## 2. The model

The 2-user MAC has two independent transmitters `X_1, X_2` and one
shared receiver `Y`:

```
   X_1  ──►  [ H_1 F_1 ]  ──┐
                            ├──►  Y = H_1 F_1 X_1 + H_2 F_2 X_2 + Z,
   X_2  ──►  [ H_2 F_2 ]  ──┘     Z ~ CN(0, σ² I_{d_Y}).
```

In DAG language this is a 3-node graph with **two roots** (`X_1, X_2`)
and one non-root sink (`Y`):

- Nodes `V_0 = X_1` and `V_1 = X_2` are the user-input roots, with
  `X_k ~ CN(0, I_{d_k})` and mutually independent.
- Node `V_2 = Y` is a non-root with parents `{V_0, V_1}` and edge
  transforms `A_{2,0} = H_1 F_1`, `A_{2,1} = H_2 F_2`.

The rate region's three facets are conditional mutual informations:

```
I_1   = I(X_1; Y | X_2)            (user-1 facet)
I_2   = I(X_2; Y | X_1)            (user-2 facet)
I_12  = I(X_1, X_2; Y)             (sum-rate facet)
```

Each one is a log-determinant difference of sub-block Schur complements
of the support covariance — exactly the formula evaluated by
`conditional_mutual_information_from_k`.

---

## 3. Compute the three pentagon CMIs

```python
import torch
from gaussian_dag_cmi import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
)

torch.manual_seed(0)
d, sigma = 2, 0.5

# Device-agnostic: same code runs on CPU or CUDA.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.complex128

# Fixed channels and identity user-input covariances.
H_1 = torch.randn(d, d, dtype=dtype, device=DEVICE)
H_2 = torch.randn(d, d, dtype=dtype, device=DEVICE)
F_1 = torch.eye(d, dtype=dtype, device=DEVICE)
F_2 = torch.eye(d, dtype=dtype, device=DEVICE)
Sigma_root = torch.eye(d, dtype=dtype, device=DEVICE)
Sigma_Z = (sigma ** 2) * torch.eye(d, dtype=dtype, device=DEVICE)

# Build the 3-node 2-root DAG: nodes {X_1, X_2, Y} with Y's parents = {X_1, X_2}.
K = compute_k_blocks_multiroot(
    num_nodes=3,
    roots=[0, 1],                                     # X_1, X_2
    parents={2: [0, 1]},                              # Y's parents
    edge_mats={(2, 0): H_1 @ F_1, (2, 1): H_2 @ F_2},
    root_covs={0: Sigma_root, 1: Sigma_root},
    noise_covs={2: Sigma_Z},
)

I_1   = conditional_mutual_information_from_k(K, A=[0],    B=[2], C=[1])
I_2   = conditional_mutual_information_from_k(K, A=[1],    B=[2], C=[0])
I_12  = conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[])

print(f"I_1  = I(X_1; Y | X_2) = {I_1.item():.4f} nats")
print(f"I_2  = I(X_2; Y | X_1) = {I_2.item():.4f} nats")
print(f"I_12 = I(X_1, X_2; Y)  = {I_12.item():.4f} nats")
```

What just happened:

- `compute_k_blocks_multiroot` propagated the root covariances through
  the DAG and produced the canonical K-blocks `K[(0,0)] = Σ_{X_1}`,
  `K[(1,1)] = Σ_{X_2}`, `K[(1,0)] = 0` (independent roots),
  `K[(2,0)], K[(2,1)], K[(2,2)]`.
- Each `conditional_mutual_information_from_k(K, A, B, C)` call read the
  K-blocks indexed by `A ∪ B ∪ C`, assembled the support covariance
  `Σ_{S,S}`, formed the two Schur complements `Σ_{A|C}` and
  `Σ_{A|BC}`, and returned `log det Σ_{A|C} − log det Σ_{A|BC}` as a
  differentiable scalar tensor.

The returned values are PyTorch tensors in **nats**, differentiable
through `H_k`, `F_k`, and the covariances — exactly what later tutorials
need for projected gradient ascent.

> **Pentagon chain rule.** By the chain rule of mutual information,
> `I_12 = I_1 + I(X_2; Y)`. Try it: compute `I(X_2; Y)` with
> `conditional_mutual_information_from_k(K, A=[1], B=[2], C=[])` and
> check that `I_1 + I(X_2; Y)` equals `I_12` within complex128
> precision. The library's test suite checks this identity in
> `tests/test_conditional_information.py::test_pentagon_chain_rule`.

---

## 4. Sanity check: scalar case matches log(1 + SNR)

For `d = 1` with `X_k ~ CN(0, P_k)` and `H_k = h_k`, the closed forms are

```
I_1  = log(1 + |h_1|^2 P_1 / σ^2)
I_2  = log(1 + |h_2|^2 P_2 / σ^2)
I_12 = log(1 + (|h_1|^2 P_1 + |h_2|^2 P_2) / σ^2).
```

Run a short check:

```python
import math
import torch
from gaussian_dag_cmi import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
)

DTYPE = torch.complex128
sigma2, P_1, P_2 = 0.7, 1.3, 0.4
h_1 = complex(0.8, -0.5)
h_2 = complex(-0.2, 0.9)

F_1 = torch.tensor([[math.sqrt(P_1)]], dtype=DTYPE)
F_2 = torch.tensor([[math.sqrt(P_2)]], dtype=DTYPE)
H_1 = torch.tensor([[h_1]], dtype=DTYPE)
H_2 = torch.tensor([[h_2]], dtype=DTYPE)

K = compute_k_blocks_multiroot(
    num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
    edge_mats={(2, 0): H_1 @ F_1, (2, 1): H_2 @ F_2},
    root_covs={0: torch.eye(1, dtype=DTYPE), 1: torch.eye(1, dtype=DTYPE)},
    noise_covs={2: sigma2 * torch.eye(1, dtype=DTYPE)},
)
I_1_lib = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1]).item()
I_1_ref = math.log(1.0 + abs(h_1) ** 2 * P_1 / sigma2)
print(f"library  : I_1 = {I_1_lib:.10f}")
print(f"classical: I_1 = {I_1_ref:.10f}")
```

The two numbers should agree to ~15 digits in IEEE double precision. The
test suite checks this and the analogous identities for `I_2` and
`I_12`, plus the MIMO log-det generalisation, in
[`tests/test_closed_form.py`](../tests/test_closed_form.py).

---

## 5. What is next?

- **Tutorial 2** opens up `compute_k_blocks_multiroot` and
  `conditional_mutual_information_from_k`: the multi-root base case,
  how K-blocks are stored, block extraction, and the Schur complement
  that turns covariances into conditional MI.
- **Tutorial 3** makes the precoders `F_1, F_2` *trainable* and runs
  projected gradient ascent on the MAC pentagon sum.
- **Tutorial 4** introduces sign-indefinite rate functions (secrecy
  rate) and the descent companion `pga_descent`.
- **Tutorial 5** is the capstone: 12-node random multi-hop MAC,
  per-relay processing under a shared budget.
