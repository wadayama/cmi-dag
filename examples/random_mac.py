"""Rate-region maximization on a random 12-node multi-hop multi-access network.

Reproduces `random_mac.pdf` of the companion paper qualitatively (same
hyperparameters, different network/channel instance — see "Reproducibility"
in `examples/README.md`). A randomly generated layered Gaussian DAG has two
source nodes s_1, s_2, three relay layers of three nodes each (nine
processing matrices in total), and a single sink t. Under a shared
total-power budget the framework jointly optimizes every relay's processing
matrix to *maximize* the MAC facet sum

    U = I(V_{s_1}; V_t | V_{s_2}) + I(V_{s_2}; V_t | V_{s_1}) + I(V_{s_1}, V_{s_2}; V_t),

the same conditional-MI objective as the small 2-user MAC example, here
applied at scale on a network with no closed-form rate region.

Usage:
    uv sync --extra examples
    uv run python examples/random_mac.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from gaussian_dag.optimize import pga_ascent
from gaussian_dag.projections import project_total_power
from gaussian_dag_cmi import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LAYERS = 5                  # 1 source layer + (L-2) relay layers + 1 sink
LAYER_WIDTH = 3                 # nodes per relay layer
D = 4                           # per-node vector dimension
EDGE_PROB = 0.6                 # layer-to-layer random connection probability
NOISE_VAR = 1.0
TOTAL_POWER = 36.0              # shared budget sum_i ||F_i||_F^2 <= P
NUM_ITERS = 800
STEP_SIZE = 0.003
SEED = 7

# Node convention: nodes 0 and 1 are the two sources; the last node is the sink;
# all nodes in between are relays carrying a processing matrix.
N_SRC = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cn_randn(*shape: int, generator: torch.Generator) -> torch.Tensor:
    """One standard complex-Gaussian sample CN(0, I): Re, Im i.i.d. N(0, 1/2)."""
    re = torch.randn(*shape, dtype=torch.float64, generator=generator)
    im = torch.randn(*shape, dtype=torch.float64, generator=generator)
    return torch.complex(re, im) / 2.0 ** 0.5


def build_random_network(
    L: int, width: int, d: int, edge_prob: float, seed: int,
):
    """Generate a random layered Gaussian DAG with two sources and one sink.

    Layer 0 holds the two source nodes s1, s2; layers 1..L-2 are relay layers
    (`width` nodes each); layer L-1 is the single sink. Each node connects to
    each node of the previous layer with probability `edge_prob`; every node
    is guaranteed at least one parent and (if non-sink) at least one child.

    Channel matrices on each edge have entries ~ CN(0, 1), drawn from the
    same Python generator used for the topology so the whole instance is
    determined by `seed`.
    """
    layers = [[0, 1]]  # two source nodes
    idx = 2
    for _ in range(L - 2):
        layers.append(list(range(idx, idx + width)))
        idx += width
    layers.append([idx])  # single sink
    M = idx + 1
    s1, s2, t = 0, 1, M - 1
    node_layer = np.zeros(M, dtype=np.int64)
    for ell, layer in enumerate(layers):
        for n in layer:
            node_layer[n] = ell

    # Random topology (NumPy RNG).
    rng = np.random.RandomState(seed)
    parents: dict[int, list[int]] = {}
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

    # Per-edge complex Gaussian channels (PyTorch RNG, separate stream).
    g = torch.Generator(device="cpu").manual_seed(seed)
    H = {
        (j, i): cn_randn(d, d, generator=g).to(DTYPE).to(DEVICE)
        for (j, i) in edges
    }
    return M, parents, edges, node_layer, s1, s2, t, H


def build_network_dag(
    F_list: list[torch.Tensor], H: dict, parents: dict, M: int,
) -> dict:
    """Multi-root K-recursion inputs for the random MAC DAG.

    Source nodes emit their signals directly; an edge out of relay node i
    carries A_{ji} = H_{ji} F_i (channel composed with relay processing).
    `F_list[i - N_SRC]` is the processing matrix of relay node i.
    """
    d = F_list[0].shape[-1]
    eye = torch.eye(d, dtype=DTYPE, device=DEVICE)
    edge_mats: dict[tuple[int, int], torch.Tensor] = {}
    for j in parents:
        for i in parents[j]:
            edge_mats[(j, i)] = (
                H[(j, i)] if i < N_SRC else H[(j, i)] @ F_list[i - N_SRC]
            )
    return dict(
        num_nodes=M,
        roots=[0, 1],
        parents=parents,
        edge_mats=edge_mats,
        root_covs={0: eye, 1: eye},
        noise_covs={j: NOISE_VAR * eye for j in range(N_SRC, M)},
    )


def mac_facets(
    F_list: list[torch.Tensor], H: dict, parents: dict, M: int,
    s1: int, s2: int, t: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three MAC rate-region facets from one K-recursion forward pass."""
    K = compute_k_blocks_multiroot(**build_network_dag(F_list, H, parents, M))
    I1 = conditional_mutual_information_from_k(K, A=[s1], B=[t], C=[s2])
    I2 = conditional_mutual_information_from_k(K, A=[s2], B=[t], C=[s1])
    I12 = conditional_mutual_information_from_k(K, A=[s1, s2], B=[t], C=[])
    return I1, I2, I12


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    here = Path(__file__).resolve().parent
    (here / "results").mkdir(parents=True, exist_ok=True)
    (here / "figures").mkdir(parents=True, exist_ok=True)

    M, parents, edges, node_layer, s1, s2, t, H = build_random_network(
        NUM_LAYERS, LAYER_WIDTH, D, EDGE_PROB, SEED
    )
    assert len({s1, s2, t}) == 3, "the two sources and the sink must be distinct"
    n_proc = M - N_SRC - 1  # one processing matrix per relay node

    # Uniform allocation: every relay identity-processing, equal power share.
    scale = (TOTAL_POWER / (n_proc * D)) ** 0.5
    F_list = [
        (scale * torch.eye(D, dtype=DTYPE, device=DEVICE)).clone().requires_grad_(True)
        for _ in range(n_proc)
    ]

    # Record (I1, I2, I12) per iteration inside the closure.
    I1_hist: list[float] = []
    I2_hist: list[float] = []
    I12_hist: list[float] = []

    def closure() -> torch.Tensor:
        I1, I2, I12 = mac_facets(F_list, H, parents, M, s1, s2, t)
        I1_hist.append(I1.item())
        I2_hist.append(I2.item())
        I12_hist.append(I12.item())
        return I1 + I2 + I12

    def projector(params: list[torch.Tensor]) -> list[torch.Tensor]:
        return project_total_power(params, TOTAL_POWER)

    U_history = pga_ascent(
        closure, F_list,
        step_size=STEP_SIZE, num_iters=NUM_ITERS, projector=projector,
    )

    # Final post-update facets and per-relay power consumption.
    with torch.no_grad():
        I1_f, I2_f, I12_f = mac_facets(F_list, H, parents, M, s1, s2, t)
    power_final = np.array(
        [float(f.detach().norm() ** 2) for f in F_list], dtype=np.float64
    )

    npz_path = here / "results" / "random_mac.npz"
    np.savez(
        npz_path,
        U_history=np.array(U_history, dtype=np.float64),
        i1_history=np.array(I1_hist, dtype=np.float64),
        i2_history=np.array(I2_hist, dtype=np.float64),
        i12_history=np.array(I12_hist, dtype=np.float64),
        facets_final=np.array(
            [I1_f.item(), I2_f.item(), I12_f.item()], dtype=np.float64,
        ),
        power_final=power_final,
        edges=np.array(edges, dtype=np.int64),
        node_layer=node_layer,
        s1=np.array(s1), s2=np.array(s2), t=np.array(t), M=np.array(M),
        config=dict(
            d=D, num_layers=NUM_LAYERS, layer_width=LAYER_WIDTH,
            edge_prob=EDGE_PROB, n_proc=n_proc,
            total_power=TOTAL_POWER, noise_var=NOISE_VAR,
            num_iters=NUM_ITERS, step_size=STEP_SIZE, seed=SEED,
        ),
    )
    print(f"results -> {npz_path}")

    # Figure: network diagram (top) and optimization trace (bottom).
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    })

    # Node positions: x = layer, y = centered position within the layer.
    layer_nodes: dict[int, list[int]] = {}
    for n in range(M):
        layer_nodes.setdefault(int(node_layer[n]), []).append(n)
    pos = {}
    for ell, nodes in layer_nodes.items():
        n_l = len(nodes)
        for k, n in enumerate(sorted(nodes)):
            pos[n] = (float(ell), k - (n_l - 1) / 2.0)

    cmap = mpl.cm.plasma
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.5, 5.0))

    for (j, i) in edges:
        axA.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                 "-", color="k", linewidth=0.5, zorder=1)
    relay = list(range(N_SRC, M - 1))
    xs = [pos[n][0] for n in relay]
    ys = [pos[n][1] for n in relay]
    sc = axA.scatter(xs, ys, c=power_final, cmap=cmap, s=70,
                     edgecolors="k", linewidths=0.5, zorder=2)
    for endpoint in (s1, s2, t):
        axA.scatter([pos[endpoint][0]], [pos[endpoint][1]], marker="s", s=70,
                    color="white", edgecolors="k", linewidths=0.8, zorder=2)
    axA.annotate(r"source $s_2$", pos[s2], textcoords="offset points",
                 xytext=(0, 14), ha="center", fontsize=7)
    axA.annotate(r"source $s_1$", pos[s1], textcoords="offset points",
                 xytext=(0, -15), va="top", ha="center", fontsize=7)
    axA.annotate(r"sink $t$", pos[t], textcoords="offset points",
                 xytext=(0, 26), ha="center", fontsize=7)
    axA.set_title("(a) random multi-hop MAC network")
    axA.set_xlabel("layer")
    axA.set_yticks([])
    axA.set_xlim(-0.5, float(node_layer.max()) + 0.5)
    cbar = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.03)
    cbar.set_label(r"node power $\|F_i\|_F^2$", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    it = np.arange(NUM_ITERS)
    axB.plot(it, I1_hist, "-", color="tab:orange", linewidth=1.0,
             label=r"$I_1=I(V_{s_1};V_t\!\mid\! V_{s_2})$")
    axB.plot(it, I2_hist, "-", color="tab:green", linewidth=1.0,
             label=r"$I_2=I(V_{s_2};V_t\!\mid\! V_{s_1})$")
    axB.plot(it, I12_hist, "-", color="tab:red", linewidth=1.0,
             label=r"$I_{12}=I(V_{s_1},V_{s_2};V_t)$")
    axB.plot(it, U_history, "-", color="tab:blue", linewidth=1.5,
             label=r"objective $U=I_1+I_2+I_{12}$")
    axB.set_xlabel("projected-gradient iteration")
    axB.set_ylabel("nats")
    axB.set_title("(b) optimization trace")
    axB.set_xlim(0, NUM_ITERS - 1)
    axB.legend(loc="center right", bbox_to_anchor=(1.0, 0.64))

    fig.tight_layout()
    fig_path = here / "figures" / "random_mac.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"figure  -> {fig_path}")


if __name__ == "__main__":
    main()
