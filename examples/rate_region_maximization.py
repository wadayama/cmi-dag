"""Rate-region maximization on a 2-user vector Gaussian MAC.

Reproduces `rate_region_evolution.pdf` of the companion paper qualitatively
(same hyperparameters, different channel realization — see "Reproducibility"
in `examples/README.md`). On a single fixed realization of a 2-user MIMO MAC,
the per-user precoders (F1, F2) are jointly optimized by projected gradient
ascent to *maximize* the linear conditional-MI objective

    U(eta) = I1 + I2 + I12,
    I1 = I(X1; Y | X2),  I2 = I(X2; Y | X1),  I12 = I(X1, X2; Y),

under a shared total-power budget. Snapshots of the rate-region pentagon
along the iterations show the achievable region expanding.

Usage:
    uv sync --extra examples
    uv run python examples/rate_region_maximization.py
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

D = 4                           # antennas per user
NOISE_VAR = 1.0                 # receiver noise variance sigma^2
TOTAL_POWER = 8.0               # shared budget sum_k ||F_k||_F^2 <= P
NUM_ITERS = 120
STEP_SIZE = 0.01
SEED = 7                        # RNG seed for the fixed channel realization
PENTAGON_DRAW_ITERS = [0, 10, 25, 55, 120]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cn_randn(*shape: int, generator: torch.Generator) -> torch.Tensor:
    """One standard complex-Gaussian sample CN(0, I): Re, Im i.i.d. N(0, 1/2)."""
    re = torch.randn(*shape, dtype=torch.float64, generator=generator)
    im = torch.randn(*shape, dtype=torch.float64, generator=generator)
    return torch.complex(re, im) / 2.0 ** 0.5


def build_mac_dag(
    F1: torch.Tensor, F2: torch.Tensor,
    H1: torch.Tensor, H2: torch.Tensor,
) -> dict:
    """Multi-root MAC DAG: two transmitter roots (X1, X2) feeding receiver Y."""
    d = F1.shape[-1]
    d_y = H1.shape[-2]
    eye_d = torch.eye(d, dtype=DTYPE, device=DEVICE)
    eye_y = torch.eye(d_y, dtype=DTYPE, device=DEVICE)
    return dict(
        num_nodes=3,
        roots=[0, 1],
        parents={2: [0, 1]},
        edge_mats={(2, 0): H1 @ F1, (2, 1): H2 @ F2},
        root_covs={0: eye_d, 1: eye_d},
        noise_covs={2: NOISE_VAR * eye_y},
    )


def pentagon_mi(
    F1: torch.Tensor, F2: torch.Tensor,
    H1: torch.Tensor, H2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three MAC-pentagon conditional MIs (I1, I2, I12)."""
    K = compute_k_blocks_multiroot(**build_mac_dag(F1, F2, H1, H2))
    I1 = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    I2 = conditional_mutual_information_from_k(K, A=[1], B=[2], C=[0])
    I12 = conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[])
    return I1, I2, I12


def pentagon_vertices(I1: float, I2: float, I12: float) -> list[tuple[float, float]]:
    """Vertices of the MAC pentagon {R1 <= I1, R2 <= I2, R1+R2 <= I12}.

    Reduces to a rectangle when the sum constraint is slack.
    """
    I1 = max(I1, 0.0)
    I2 = max(I2, 0.0)
    I12 = max(I12, 0.0)
    if I12 >= I1 + I2:
        return [(0.0, 0.0), (I1, 0.0), (I1, I2), (0.0, I2)]
    v3y = max(0.0, min(I2, I12 - I1))
    v4x = max(0.0, min(I1, I12 - I2))
    return [(0.0, 0.0), (I1, 0.0), (I1, v3y), (v4x, I2), (0.0, I2)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    here = Path(__file__).resolve().parent
    (here / "results").mkdir(parents=True, exist_ok=True)
    (here / "figures").mkdir(parents=True, exist_ok=True)

    # Fixed MIMO MAC channel realization, entries ~ CN(0, 1).
    g = torch.Generator(device="cpu").manual_seed(SEED)
    H1 = cn_randn(D, D, generator=g).to(DTYPE).to(DEVICE)
    H2 = cn_randn(D, D, generator=g).to(DTYPE).to(DEVICE)

    # Uniform equal-split initialization: each ||F_k||_F^2 = P/2, sum = P.
    scale = (TOTAL_POWER / (2.0 * D)) ** 0.5
    eye = torch.eye(D, dtype=DTYPE, device=DEVICE)
    F1 = (scale * eye).clone().requires_grad_(True)
    F2 = (scale * eye).clone().requires_grad_(True)

    # PGA on U = I1 + I2 + I12 under the shared Frobenius-budget projector.
    # Record (I1, I2, I12) inside the closure so each iteration's pre-update
    # facet values are captured alongside the U history.
    I1_hist: list[float] = []
    I2_hist: list[float] = []
    I12_hist: list[float] = []

    def closure() -> torch.Tensor:
        I1, I2, I12 = pentagon_mi(F1, F2, H1, H2)
        I1_hist.append(I1.item())
        I2_hist.append(I2.item())
        I12_hist.append(I12.item())
        return I1 + I2 + I12

    def projector(params: list[torch.Tensor]) -> list[torch.Tensor]:
        return project_total_power(params, TOTAL_POWER)

    U_history = pga_ascent(
        closure, [F1, F2],
        step_size=STEP_SIZE, num_iters=NUM_ITERS, projector=projector,
    )

    # Final post-update snapshot so the trajectory has NUM_ITERS + 1 boundaries.
    with torch.no_grad():
        I1_f, I2_f, I12_f = pentagon_mi(F1, F2, H1, H2)
    I1_hist.append(I1_f.item())
    I2_hist.append(I2_f.item())
    I12_hist.append(I12_f.item())

    snapshot_iters = np.arange(NUM_ITERS + 1, dtype=np.int64)
    I1_arr = np.array(I1_hist)
    I2_arr = np.array(I2_hist)
    I12_arr = np.array(I12_hist)
    U_arr = I1_arr + I2_arr + I12_arr

    npz_path = here / "results" / "rate_region_maximization.npz"
    np.savez(
        npz_path,
        snapshot_iters=snapshot_iters,
        I1_traj=I1_arr, I2_traj=I2_arr, I12_traj=I12_arr,
        U_history=np.array(U_history, dtype=np.float64),
        H1=H1.detach().cpu().numpy(),
        H2=H2.detach().cpu().numpy(),
        F1_star=F1.detach().cpu().numpy(),
        F2_star=F2.detach().cpu().numpy(),
        config=dict(
            d=D, total_power=TOTAL_POWER, noise_var=NOISE_VAR,
            num_iters=NUM_ITERS, step_size=STEP_SIZE, seed=SEED,
            pentagon_draw_iters=PENTAGON_DRAW_ITERS,
        ),
    )
    print(f"results -> {npz_path}")

    # Figure: pentagon snapshots (top) and objective trajectory (bottom).
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    })

    draw_iters = [int(t) for t in PENTAGON_DRAW_ITERS]
    dmax = float(max(draw_iters)) if max(draw_iters) > 0 else 1.0
    cmap = mpl.cm.plasma

    def iter_color(it: int):
        return cmap(0.05 + 0.67 * (it / dmax))

    def closed(verts: list[tuple[float, float]]):
        return ([v[0] for v in verts] + [verts[0][0]],
                [v[1] for v in verts] + [verts[0][1]])

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.5, 4.7))

    for it in draw_iters:
        xs, ys = closed(pentagon_vertices(
            float(I1_arr[it]), float(I2_arr[it]), float(I12_arr[it])))
        axA.plot(xs, ys, "-", color=iter_color(it), linewidth=0.8,
                 label=f"iter {it}")
    axA.set_xlabel(r"$R_1$ (nats)")
    axA.set_ylabel(r"$R_2$ (nats)")
    axA.set_title("(a) rate region along optimization")
    axA.set_xlim(0.0, axA.get_xlim()[1])
    axA.set_ylim(0.0, axA.get_ylim()[1])
    axA.legend(loc="lower left", framealpha=0.9)

    axB.plot(snapshot_iters, U_arr, "-", color="0.55", linewidth=0.8, zorder=1)
    for it in draw_iters:
        axB.plot(it, U_arr[it], "o", color=iter_color(it),
                 markersize=4.5, markeredgecolor="k", markeredgewidth=0.4,
                 zorder=2)
    axB.set_xlabel("projected-gradient iteration")
    axB.set_ylabel(r"$U = I_1 + I_2 + I_{12}$ (nats)")
    axB.set_title("(b) objective")
    axB.set_xlim(0, NUM_ITERS)

    fig.tight_layout()
    fig_path = here / "figures" / "rate_region_maximization.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"figure  -> {fig_path}")


if __name__ == "__main__":
    main()
