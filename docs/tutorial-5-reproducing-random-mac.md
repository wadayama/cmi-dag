# Tutorial 5 — Reproducing the random multi-hop MAC figure

This capstone tutorial walks through the random multi-hop
multiple-access network experiment of the companion paper: 12 nodes
spanning 5 layers (2 source nodes + 3 relay layers of 3 nodes each + 1
sink), per-node dimension `d = 4`. Each of the 9 relays carries a
controllable processing matrix `F_i`, shared across that relay's
outgoing edges. Projected gradient ascent jointly optimises
`{F_i}_{i=1}^{9}` under a shared total power budget `P = 36`. The MAC
facet sum

```
   U  =  I(V_{s_1}; V_t | V_{s_2})  +  I(V_{s_2}; V_t | V_{s_1})
                                    +  I(V_{s_1}, V_{s_2}; V_t)
```

rises monotonically from its uniform-initial value to its
optimised value — the figure of merit of the experiment.

All of this is implemented as
[`examples/random_mac.py`](../examples/random_mac.py). This tutorial
explains what is in that script piece by piece. To run it end-to-end at
any time:

```bash
uv sync --extra examples
uv run python examples/random_mac.py
```

The script writes `examples/results/random_mac.npz` and
`examples/figures/random_mac.pdf`.

![random multi-hop MAC](figures/random_mac.png)

*Figure: panel (a) shows the random multi-hop network with relay nodes
shaded by their final per-node power `||F_i||_F^2`. Panel (b) shows the
optimisation trajectory of the three MAC pentagon facets and their sum
`U`.*

---

## 1. Build the random layered topology

The DAG has layers indexed `0, 1, ..., L-1`:

- Layer 0 holds the two source nodes `s_1, s_2`.
- Layers 1 through `L-2` are *relay* layers, each containing
  `LAYER_WIDTH` nodes.
- Layer `L-1` holds the single sink `t`.

Each node in layer `ell >= 1` connects to each node of layer `ell - 1`
with probability `EDGE_PROB`, with two safety nets:

- Every node must have at least one parent.
- Every non-sink node must have at least one child.

Per-edge channels are independent `CN(0, 1)` matrices of shape
`d × d`. Both the topology and the channels are deterministic in a
single integer `SEED`:

```python
import numpy as np
import torch

DTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LAYERS  = 5
LAYER_WIDTH = 3
D           = 4
EDGE_PROB   = 0.6
SEED        = 7
N_SRC       = 2                                    # nodes 0, 1 are sources

def cn_randn(*shape, generator):
    """One standard complex-Gaussian sample CN(0, I): Re, Im i.i.d. N(0, 1/2)."""
    re = torch.randn(*shape, dtype=torch.float64, generator=generator)
    im = torch.randn(*shape, dtype=torch.float64, generator=generator)
    return torch.complex(re, im) / 2.0 ** 0.5

def build_random_network(L, width, d, edge_prob, seed):
    layers = [[0, 1]]
    idx = 2
    for _ in range(L - 2):
        layers.append(list(range(idx, idx + width)))
        idx += width
    layers.append([idx])
    M = idx + 1
    s_1, s_2, t = 0, 1, M - 1

    rng = np.random.RandomState(seed)
    parents = {}
    for ell in range(1, L):
        prev, cur = layers[ell - 1], layers[ell]
        for n in cur:
            mask = rng.rand(len(prev)) < edge_prob
            if not mask.any():
                mask[rng.randint(len(prev))] = True
            parents[n] = [prev[k] for k in range(len(prev)) if mask[k]]
        for p in prev:
            if not any(p in parents[n] for n in cur):
                parents[cur[rng.randint(len(cur))]].append(p)
    for n in parents:
        parents[n] = sorted(set(parents[n]))
    edges = [(j, i) for j in sorted(parents) for i in parents[j]]

    g = torch.Generator(device="cpu").manual_seed(seed)
    H = {(j, i): cn_randn(d, d, generator=g).to(DTYPE).to(DEVICE)
         for (j, i) in edges}
    return M, parents, edges, s_1, s_2, t, H

M, parents, edges, s_1, s_2, t, H = build_random_network(
    NUM_LAYERS, LAYER_WIDTH, D, EDGE_PROB, SEED
)
print(f"M = {M} nodes, |E| = {len(edges)} edges, "
      f"sources = ({s_1}, {s_2}), sink = {t}")
```

For `SEED = 7` you should see `M = 12`, two sources at nodes 0 and 1,
and a single sink at node 11, with 9 relay nodes in between.

---

## 2. Allocate the controllable processing matrices

The sources emit isotropic signals; only relays carry processing
matrices. With `n_proc = 9` relays and `d = 4`, the budget `P = 36` is
chosen so that the uniform initialisation

```
F_i = sqrt(P / (n_proc · d)) · I_d  =  sqrt(1) · I_d  =  I_d
```

corresponds to *identity processing* at every relay. This is a natural
baseline against which the optimisation improvement is measured.

```python
P = 36.0
n_proc = M - N_SRC - 1                              # 9
scale = (P / (n_proc * D)) ** 0.5                   # = 1.0
F_list = [
    (scale * torch.eye(D, dtype=DTYPE, device=DEVICE)).clone().requires_grad_(True)
    for _ in range(n_proc)
]
```

The list `F_list[0], ..., F_list[8]` corresponds to relay nodes
`N_SRC, N_SRC + 1, ..., M - 2` (the relay nodes are numbered after the
sources).

---

## 3. Compose channels with relay processing

For each edge `(j, i)`:

- If `i < N_SRC` the edge leaves a source, and `A_{j,i} = H_{j,i}` —
  the channel directly.
- If `i >= N_SRC` the edge leaves a relay, and
  `A_{j,i} = H_{j,i} · F_{i - N_SRC}` — the channel composed with the
  relay's *shared* processing.

```python
from cmi_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
)

def build_dag(F_list):
    eye = torch.eye(D, dtype=DTYPE, device=DEVICE)
    edge_mats = {}
    for j in parents:
        for i in parents[j]:
            edge_mats[(j, i)] = (
                H[(j, i)] if i < N_SRC
                else H[(j, i)] @ F_list[i - N_SRC]
            )
    return dict(
        num_nodes=M,
        roots=[0, 1],
        parents=parents,
        edge_mats=edge_mats,
        root_covs={0: eye, 1: eye},
        noise_covs={j: eye for j in range(N_SRC, M)},
    )
```

PyTorch's reverse-mode AD will accumulate gradient contributions from
every edge that uses `F_i` into a single `F_list[i - N_SRC].grad` — no
special handling required (Tutorial 4 of the parent library covers this
mechanism).

---

## 4. The MAC facet sum at the sink

The objective is the same three-CMI sum as Tutorial 3, but now
evaluated between the two source nodes and the *sink* at the far end
of the network:

```python
def mac_facets(F_list):
    K = compute_k_blocks_multiroot(**build_dag(F_list))
    I_1  = conditional_mutual_information_from_k(K, A=[s_1],      B=[t], C=[s_2])
    I_2  = conditional_mutual_information_from_k(K, A=[s_2],      B=[t], C=[s_1])
    I_12 = conditional_mutual_information_from_k(K, A=[s_1, s_2], B=[t], C=[])
    return I_1, I_2, I_12

def compute_U():
    I_1, I_2, I_12 = mac_facets(F_list)
    return I_1 + I_2 + I_12

with torch.no_grad():
    I_1, I_2, I_12 = mac_facets(F_list)
print(f"uniform init: I_1 = {I_1.item():.3f}, "
      f"I_2 = {I_2.item():.3f}, "
      f"I_12 = {I_12.item():.3f}, "
      f"U = {(I_1 + I_2 + I_12).item():.3f}")
```

One forward sweep of `compute_k_blocks_multiroot` propagates every
node-pair covariance along the DAG;
`conditional_mutual_information_from_k` then reads the required
sub-block Schur complements three times and returns the
differentiable scalars.

---

## 5. PGA with a shared-budget projector

The 9 matrices share a single total power budget. We use
`project_total_power`:

```python
from gaussian_dag.optimize import pga_ascent
from gaussian_dag.projections import project_total_power

NUM_ITERS = 800
STEP_SIZE = 0.003

def projector(params):
    return project_total_power(params, P)

history = pga_ascent(
    compute_U, F_list,
    step_size=STEP_SIZE, num_iters=NUM_ITERS, projector=projector,
)

with torch.no_grad():
    I_1_f, I_2_f, I_12_f = mac_facets(F_list)
power_final = [float(f.detach().norm() ** 2) for f in F_list]

print(f"U: {history[0]:.4f} -> {history[-1]:.4f} nats")
print(f"optimised facets: I_1 = {I_1_f.item():.3f}, "
      f"I_2 = {I_2_f.item():.3f}, "
      f"I_12 = {I_12_f.item():.3f}")
print(f"total power = {sum(power_final):.4f}  (budget {P:.1f})")
print(f"per-relay power: min = {min(power_final):.3f}, "
      f"max = {max(power_final):.3f}, "
      f"uniform share = {P / n_proc:.3f}")
```

A single common scale factor is applied to *every* `F_i`, preserving
the relative magnitudes that the gradient has discovered. The optimised
per-node powers are non-uniform: PGA reshapes the shared budget across
relays based on where in the network the marginal gain is largest.

---

## 6. Rendering the figure

The full script
[`examples/random_mac.py`](../examples/random_mac.py) records `(I_1,
I_2, I_12)` per iteration inside the closure, then renders a 2-panel
PDF: panel (a) is the network diagram with relay nodes colour-coded by
their final power, panel (b) is the per-facet and total optimisation
trajectory. This tutorial intentionally stops at the numerical
optimisation step — you can either copy the matplotlib block from the
example script, or just run the script directly:

```bash
uv run python examples/random_mac.py
```

and inspect `examples/figures/random_mac.pdf`.

---

## 7. Where to go from here

- Edit the constants at the top of
  [`examples/random_mac.py`](../examples/random_mac.py) — larger
  `NUM_LAYERS`, denser `EDGE_PROB`, different `SEED` — and rerun to see
  how the optimised allocation changes.
- Replace the MAC facet sum with a different sign-indefinite rate
  function (e.g. a Han–Kobayashi-style facet with mixed
  positive/negative coefficients) by editing the `Summand` list passed
  to `evaluate_rate_functions`.
- Apply the same template to a topology of your own: define `parents`,
  build the per-edge channels `H`, decide on the parameter-sharing
  pattern, pick a rate function, choose a projector, and call
  `pga_ascent` or `pga_descent` depending on the sign of the natural
  objective.
- For a more thorough correctness story, browse
  [`tests/test_closed_form.py`](../tests/test_closed_form.py): the
  3-facet MAC pentagon CMIs from the multi-root K-recursion + Schur
  complement pipeline agree exactly with the classical log-det
  capacity formulas.

Congratulations — you have walked through the full `cmi-dag`
pipeline, from a 2-user MAC sanity check to a 12-node random multi-hop
network optimised under a shared budget.
