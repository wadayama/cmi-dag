# Tutorial 4 — Sign-indefinite objectives and `pga_descent`

In Tutorial 3 every coefficient `α_{T,n}` of every rate function was
positive: we maximised a *sum* of conditional MIs. Many real
multi-terminal design problems are not sums — they are *differences*.
Secrecy rate on a wiretap channel is the canonical example:

```
   U_sec(F)  =  I(X; Y)  −  I(X; Z),
```

where `Y` is the legitimate receiver and `Z` is an eavesdropper. The
rate-function machinery of Tutorial 3 covers this case directly: just
let the second summand carry the coefficient `α = −1`.

This tutorial also introduces `pga_descent`, the descent companion of
the parent library's `pga_ascent`. Even though the wiretap example is
naturally maximised (the secrecy rate is the *good* quantity, we
ascend it), there are application-side objectives — outage probability,
distortion, leakage — that are naturally *minimised*. `pga_descent`
exists for that case.

By the end you will understand:

- How to express a sign-indefinite rate function with `Summand`
  coefficients of either sign.
- The wiretap DAG: one root `X`, two non-root sinks `Y` and `Z`.
- The signature symmetry between `pga_ascent` and `pga_descent` and
  when each one is the natural choice.

The polished end-to-end version of this experiment lives in
[`examples/secure_precoding.py`](../examples/secure_precoding.py).

---

## 1. Wiretap as a 3-node DAG

The wiretap channel has one transmitter and two receivers:

```
                 ┌──►  V_1 = Y  =  H_Y F X + Z_Y     (legitimate)
   V_0 = X ──────┤
                 └──►  V_2 = Z  =  H_Z F X + Z_Z     (eavesdropper)
```

In DAG language `V_0 = X` is the unique root with `X ~ CN(0, I_d)`,
and `V_1 = Y`, `V_2 = Z` are both non-root sinks each having `V_0` as
their only parent. Two edges leave the root, both carrying the *same*
controllable precoder `F`:

```
   A_{1, 0}  =  H_Y · F          (V_0 -> V_1)
   A_{2, 0}  =  H_Z · F          (V_0 -> V_2).
```

The precoder is shared because physically the transmitter sends a
single transmit signal that both receivers observe — the parameter
sharing pattern of the parent library's Tutorial 4 applies here too.

The secrecy rate is the difference of two unconditional MIs:

```
   U_sec(F)  =  I(X; Y)  −  I(X; Z).
```

Driving `I(X; Y)` up while pushing `I(X; Z)` down is non-trivial:
unlike pure power allocation, it requires *shaping* the precoder's
subspace away from the eavesdropper's channel.

---

## 2. Express the secrecy rate as a rate function

The rate-function machinery of Tutorial 3 covers this case directly:

```python
import torch
from cmi_dag import (
    compute_k_blocks_multiroot,
    evaluate_rate_functions,
)

torch.manual_seed(0)
d, d_eve = 4, 4
DTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

H_Y = torch.randn(d,     d, dtype=DTYPE, device=DEVICE)
H_Z = torch.randn(d_eve, d, dtype=DTYPE, device=DEVICE)

# The secrecy rate as a single rate function with TWO summands:
#   U_sec = (+1) * I(X; Y)  +  (-1) * I(X; Z).
secrecy_rate = [
    [(+1.0, [0], [1], []),                    # I(X; Y)   — legitimate
     (-1.0, [0], [2], [])],                   # I(X; Z)   — eavesdropper
]
```

The list `secrecy_rate` is a family with one rate function; that
function carries two `Summand` tuples whose coefficients are `+1` and
`−1`. `evaluate_rate_functions` returns the difference verbatim.

The framework imposes no sign constraint on `α_{T,n}` (see
[`cmi_dag/rate_region.py`](../cmi_dag/rate_region.py)):
the projected-gradient method needs only differentiability of the rate
function in the precoder, not concavity or monotonicity.

---

## 3. Ascend the secrecy rate

We use `pga_ascent` from the parent library together with
`project_frobenius_ball` (the precoder is a single matrix, so a single
Frobenius ball is the natural budget):

```python
from gaussian_dag.optimize import pga_ascent
from gaussian_dag.projections import project_frobenius_ball

P = 8.0                                       # ||F||_F^2 <= P
F = (((P / d) ** 0.5) * torch.eye(d, dtype=DTYPE, device=DEVICE)).clone()
F = F.requires_grad_(True)
Sigma_X  = torch.eye(d,     dtype=DTYPE, device=DEVICE)
Sigma_ZY = torch.eye(d,     dtype=DTYPE, device=DEVICE)
Sigma_ZZ = torch.eye(d_eve, dtype=DTYPE, device=DEVICE)

def compute_secrecy():
    K = compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0],
        parents={1: [0], 2: [0]},
        edge_mats={(1, 0): H_Y @ F, (2, 0): H_Z @ F},
        root_covs={0: Sigma_X},
        noise_covs={1: Sigma_ZY, 2: Sigma_ZZ},
    )
    return evaluate_rate_functions(K, secrecy_rate)[0]

def projector(params):
    return [project_frobenius_ball(p, P) for p in params]

history = pga_ascent(
    compute_secrecy, [F], step_size=0.02, num_iters=120, projector=projector,
)
print(f"secrecy: {history[0]:.4f} -> {history[-1]:.4f} nats")
```

The optimised `F` shapes the transmitted subspace toward the
legitimate channel's column space while reducing its overlap with the
eavesdropper's column space — without changing the total transmit
power. The two MIs separate over the iterations, which is the
qualitative point of Figure (a) in
[`examples/secure_precoding.py`](../examples/secure_precoding.py).

---

## 4. When to use `pga_descent`

`pga_ascent` maximises whatever scalar the closure returns. If your
natural objective is a *cost*, you have two equivalent options:

**(a) Maximise the negative.** Define `compute_obj()` to return
`-cost`, then call `pga_ascent`. The history this returns is the
ascending negative-cost trajectory; you can flip its sign for display.

**(b) Use `pga_descent`.** Same signature as `pga_ascent`, but the
returned history is in the **true sign** of your cost and is
monotonically non-increasing for a successful descent.

```python
from cmi_dag import pga_descent

# Suppose you want to MINIMISE the eavesdropper information I(X; Z) on its own,
# with no positive counterpart. Express it as a (single-summand) rate function
# and descend.
leakage_rate = [
    [(+1.0, [0], [2], [])],                   # f_T = I(X; Z)
]

def compute_leakage():
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0], parents={1: [0], 2: [0]},
        edge_mats={(1, 0): H_Y @ F, (2, 0): H_Z @ F},
        root_covs={0: Sigma_X},
        noise_covs={1: Sigma_ZY, 2: Sigma_ZZ},
    )
    return evaluate_rate_functions(K, leakage_rate)[0]

descent_history = pga_descent(
    compute_leakage, [F], step_size=0.02, num_iters=10, projector=projector,
)
print(f"I(X; Z): {descent_history[0]:.4f} -> {descent_history[-1]:.4f}")
```

`pga_descent` internally negates the closure, forwards to
`pga_ascent`, and flips the returned history's sign — so its API and
contract match `pga_ascent` exactly except for the optimisation
direction. The two functions live in different libraries:

```
   gaussian_dag.optimize.pga_ascent           (parent: ascend)
   cmi_dag.optimize.pga_descent      (child:  descend)
```

Pick whichever matches the *natural* sign of your objective so the
returned history reads correctly without extra bookkeeping.

> **Two ways to spell the same thing.** Minimising `cost(F)` via
> `pga_descent` is mathematically identical to maximising `−cost(F)`
> via `pga_ascent`. The difference is purely about which sign the
> history reads in. Pick the spelling that matches what your
> downstream code expects.

---

## 5. What is next?

- **Tutorial 5** is the capstone: a 12-node random multi-hop
  multi-source MAC with nine controllable relay processing matrices,
  jointly optimised under a shared total-power budget. It synthesises
  every concept of Tutorials 1–4 on a non-trivial topology.
