# Tutorial 2 — Multi-root K-recursion and conditional covariance

Tutorial 1 used `compute_k_blocks_multiroot` as a black box. This tutorial
opens the box: we look at the multi-root base case, the recursion for
non-root nodes, how K-blocks are stored, and how
`conditional_mutual_information_from_k` turns those blocks into
conditional MI via block extraction and the Schur complement.

By the end you will understand:

- Why the K-recursion needs a base case at the roots, and what it is.
- How `compute_k_blocks_multiroot` propagates covariances through
  non-root nodes.
- How `conditional_mutual_information_from_k` assembles a support
  covariance `Σ_{S,S}` and forms two Schur complements.
- The log-determinant formula
  `I(V_A; V_B | V_C) = log det Σ_{A|C} − log det Σ_{A|BC}`.

We continue with the 2-user MAC DAG of Tutorial 1; the same numerical
example now serves as a window onto what is happening under the hood.

---

## 1. Why multi-root

In single-source DAGs (the case covered by the parent library
`gaussian-dag`), node `V_0` is the unique root and carries the channel
input. Multi-terminal channels naturally have several *independent*
source nodes — two transmitters of a MAC, K transmitters of a K-user
interference channel, the private and common parts of a Han–Kobayashi
split, two sources of a multi-source relay network, and so on.
`cmi-dag` extends the K-recursion to handle any number of such
roots:

```
roots = {0, 1, ..., K-1}            # mutually independent inputs
```

with one covariance `Σ_r` per root, and `K_{r, r'} = 0` for any two
distinct roots. Non-root nodes obey the same structural equation as in
the parent library:

```
V_j = sum_{i in Pa(j)} A_{j,i} V_i + Z_j,    j ∉ roots,
                                              Z_j ~ CN(0, Σ_j),
```

with the noise vectors `Z_j` mutually independent and also independent
of the user inputs.

---

## 2. The K-recursion equations

The K-recursion fills the canonical block dictionary `K`, with one entry
per `(j, k)` such that `j >= k`. The blocks of interest are:

```
                    ┌──── j, k in roots ────┐
   K_{r, r}  = Σ_r                                    (base, self block)
   K_{r, r'} = 0     for distinct roots r > r'         (base, cross)

                    ┌──── j ∉ roots ────┐
   K_{j, k}  = sum_{i in Pa(j)} A_{j,i} K_{i, k}      (cross, k < j)
   K_{j, j}  = sum_{i,i' in Pa(j)} A_{j,i} K_{i,i'} A_{j,i'}^H + Σ_j   (self)
```

The base case at the roots seeds the recursion; the non-root equations
then walk through the DAG in topological order. Each non-root self
block `K_{j, j}` involves the **parent cross-covariance** `K_{i, i'}`,
which captures correlations introduced by shared upstream signals — for
the 2-user MAC this is what couples `K_{Y, Y}` to both transmitters'
contributions.

In ASCII, the MAC DAG looks like

```
                ┌──► A_{2,0} = H_1 F_1
   V_0 = X_1 ──┘                       ┐
                                        ├──► V_2 = Y
   V_1 = X_2 ──┐                       ┘
                └──► A_{2,1} = H_2 F_2
```

and the recursion fills

```
K[(0,0)] = Σ_{X_1}            ─┐
K[(1,1)] = Σ_{X_2}             │ base: roots
K[(1,0)] = 0                  ─┘

K[(2,0)] = A_{2,0} K[(0,0)] + A_{2,1} K[(1,0)]    = A_{2,0} Σ_{X_1}
K[(2,1)] = A_{2,0} K[(0,1)] + A_{2,1} K[(1,1)]    = A_{2,1} Σ_{X_2}
K[(2,2)] = A_{2,0} Σ_{X_1} A_{2,0}^H
          + A_{2,1} Σ_{X_2} A_{2,1}^H + Σ_Z
```

— exactly the receive-side covariance that the closed-form MAC capacity
formula uses.

> **Storage convention.** `K[(j, k)]` is stored only for `j >= k` (the
> lower-triangular half of the block matrix). To access `K_{a, b}` for
> `a < b`, use `get_K(K, a, b)` from the parent library, which applies
> the Hermitian flip `K_{a, b} = K_{b, a}^H` for you. The library's
> internals (and the next section) use this accessor throughout.

---

## 3. Reading K-blocks

K-blocks are PyTorch tensors with the natural shapes:

```python
import torch
from cmi_dag import compute_k_blocks_multiroot
from gaussian_dag import get_K                       # the parent's accessor

torch.manual_seed(0)
d = 2
DTYPE = torch.complex128

H_1 = torch.randn(d, d, dtype=DTYPE)
H_2 = torch.randn(d, d, dtype=DTYPE)
F_1 = torch.eye(d, dtype=DTYPE)
F_2 = torch.eye(d, dtype=DTYPE)

K = compute_k_blocks_multiroot(
    num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
    edge_mats={(2, 0): H_1 @ F_1, (2, 1): H_2 @ F_2},
    root_covs={0: torch.eye(d, dtype=DTYPE), 1: torch.eye(d, dtype=DTYPE)},
    noise_covs={2: torch.eye(d, dtype=DTYPE)},
)

for key, block in sorted(K.items()):
    print(f"K[{key}].shape = {tuple(block.shape)}")
print(f"||K[(1,0)]||_F = {torch.linalg.norm(K[(1, 0)]).item():.3e}  "
      "(zero: independent roots)")
print(f"K[(0,1)] via get_K = K[(1,0)]^H, "
      f"||·||_F = {torch.linalg.norm(get_K(K, 0, 1)).item():.3e}")
```

The dtype and device of every K-block match the inputs, with no
side-allocations elsewhere — the library is fully device-agnostic. The
off-diagonal root block `K[(1, 0)]` is exactly the zero tensor; the
Hermitian-flipped accessor returns the matching upper-triangular view.

---

## 4. Block extraction and the Schur complement

Conditional mutual information between three disjoint node subsets
`A, B, C ⊂ V` follows the log-determinant formula

```
I(V_A; V_B | V_C)  =  log det Σ_{A|C}  −  log det Σ_{A|BC},
```

where `Σ_{A|X}` is the conditional covariance of `V_A` given `V_X`,
obtained as the Schur complement of `Σ_{X, X}` in `Σ_{A∪X, A∪X}`:

```
Σ_{A|X}  =  Σ_{A, A}  −  Σ_{A, X} Σ_{X, X}^{−1} Σ_{X, A}.
```

`conditional_mutual_information_from_k` evaluates this by *block
extraction* — stacking the K-blocks indexed by the support
`S = A ∪ B ∪ C`. For the MAC pentagon's `I_1 = I(X_1; Y | X_2)` we have
`A = {0}`, `B = {2}`, `C = {1}`, so

```
Σ_{S,S}  =  ⎡ K_{X_1, X_1}   K_{X_1, X_2}   K_{X_1, Y}  ⎤
            ⎢ K_{X_2, X_1}   K_{X_2, X_2}   K_{X_2, Y}  ⎥
            ⎣ K_{Y,   X_1}   K_{Y,   X_2}   K_{Y,   Y}  ⎦
```

is assembled from the K-blocks (using the Hermitian flip for the upper
triangle), and the two Schur complements `Σ_{X_1 | X_2}` and
`Σ_{X_1 | X_2, Y}` are extracted from sub-blocks of `Σ_{S,S}` and fed
into `log det` — once each.

No explicit matrix inversion is formed: the Schur complement uses
`torch.linalg.solve` and the log-determinants use the parent library's
Cholesky-based `logdet_hpd`. Everything is differentiable through the
edge matrices and the covariances.

---

## 5. Try it: a non-trivial chain-rule identity

The mutual information chain rule says

```
I(A; B, C)  =  I(A; B)  +  I(A; C | B).
```

For the MAC, take `A = {X_1}`, `B = {Y}`, `C = {X_2}`. The first
right-hand-side term is `I(X_1; Y)`, which is *not* zero. The second
term is `I(X_1; X_2 | Y)` — also non-zero, even though `I(X_1; X_2) = 0`
(the two transmitters are independent), because observing `Y` couples
them:

```python
I_X1_Y_X2     = conditional_mutual_information_from_k(K, A=[0], B=[1, 2], C=[])
I_X1_Y        = conditional_mutual_information_from_k(K, A=[0], B=[2],    C=[])
I_X1_X2_givY  = conditional_mutual_information_from_k(K, A=[0], B=[1],    C=[2])
I_X1_X2       = conditional_mutual_information_from_k(K, A=[0], B=[1],    C=[])

print(f"I(X_1; Y, X_2)            = {I_X1_Y_X2.item():.6f}")
print(f"I(X_1; Y) + I(X_1; X_2|Y) = {(I_X1_Y + I_X1_X2_givY).item():.6f}")
print(f"chain-rule residual       = "
      f"{(I_X1_Y_X2 - I_X1_Y - I_X1_X2_givY).item():.3e}  (≈ 0)")
print(f"I(X_1; X_2) [independent] = {I_X1_X2.item():.3e}  (≈ 0)")
```

The chain-rule residual should be zero to within complex128 precision,
and `I(X_1; X_2)` should be exactly zero (modulo Cholesky round-off).
The library's test suite formalises both checks in
[`tests/test_conditional_information.py`](../tests/test_conditional_information.py)
and the closed-form scalar / MIMO log-det identities in
[`tests/test_closed_form.py`](../tests/test_closed_form.py).

---

## 6. What is next?

- **Tutorial 3** turns the precoders `F_1, F_2` of this MAC into
  *trainable* tensors and runs projected gradient ascent on
  `U = I_1 + I_2 + I_12` under a shared total-power budget.
- **Tutorial 4** introduces sign-indefinite rate functions (secrecy
  rate on a wiretap channel) and the descent companion `pga_descent`.
- **Tutorial 5** is the capstone: 12-node random multi-hop MAC.
