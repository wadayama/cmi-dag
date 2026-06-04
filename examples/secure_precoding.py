"""Secure precoding on a MIMO wiretap channel.

Reproduces `secure_precoding.pdf` of the companion paper qualitatively
(same hyperparameters, different channel realization — see "Reproducibility"
in `examples/README.md`). A transmitter X sends to a legitimate receiver Y
while an eavesdropper Z also observes the transmission. The precoder F is
optimized by projected gradient ascent to *maximize* the secrecy rate

    U(F) = I(X; Y) - I(X; Z),

a sign-indefinite (leakage-penalized) conditional-MI objective. I(X;Y) rises
while I(X;Z) is driven down: the divergence shows the framework optimizes a
genuine conditional-MI objective, not merely allocating power.

Usage:
    uv sync --extra examples
    uv run python examples/secure_precoding.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from gaussian_dag.optimize import pga_ascent
from gaussian_dag.projections import project_frobenius_ball
from cmi_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DTYPE = torch.complex128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = 4                           # antennas at the transmitter X and receiver Y
D_EVE = 4                       # antennas at the eavesdropper Z
NOISE_VAR = 1.0                 # receiver noise variance sigma^2
TOTAL_POWER = 8.0               # precoder power budget ||F||_F^2 <= P
NUM_ITERS = 120
STEP_SIZE = 0.02
SEED = 7                        # RNG seed for the fixed channel realization

# Wiretap DAG node convention: X = root, Y = legitimate receiver, Z = eavesdropper.
X_NODE, Y_NODE, Z_NODE = 0, 1, 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cn_randn(*shape: int, generator: torch.Generator) -> torch.Tensor:
    """One standard complex-Gaussian sample CN(0, I): Re, Im i.i.d. N(0, 1/2)."""
    re = torch.randn(*shape, dtype=torch.float64, generator=generator)
    im = torch.randn(*shape, dtype=torch.float64, generator=generator)
    return torch.complex(re, im) / 2.0 ** 0.5


def build_wiretap_dag(
    F: torch.Tensor, H_Y: torch.Tensor, H_Z: torch.Tensor,
) -> dict:
    """Wiretap DAG: a single root X feeds Y and Z through the precoder F.

    V_Y = H_Y F V_X + Z_Y,  V_Z = H_Z F V_X + Z_Z.
    """
    d_x = F.shape[-1]
    d_y = H_Y.shape[-2]
    d_z = H_Z.shape[-2]
    return dict(
        num_nodes=3,
        roots=[X_NODE],
        parents={Y_NODE: [X_NODE], Z_NODE: [X_NODE]},
        edge_mats={
            (Y_NODE, X_NODE): H_Y @ F,
            (Z_NODE, X_NODE): H_Z @ F,
        },
        root_covs={X_NODE: torch.eye(d_x, dtype=DTYPE, device=DEVICE)},
        noise_covs={
            Y_NODE: NOISE_VAR * torch.eye(d_y, dtype=DTYPE, device=DEVICE),
            Z_NODE: NOISE_VAR * torch.eye(d_z, dtype=DTYPE, device=DEVICE),
        },
    )


def wiretap_mi(
    F: torch.Tensor, H_Y: torch.Tensor, H_Z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The legitimate and eavesdropper mutual informations (I(X;Y), I(X;Z))."""
    K = compute_k_blocks_multiroot(**build_wiretap_dag(F, H_Y, H_Z))
    I_XY = conditional_mutual_information_from_k(K, A=[X_NODE], B=[Y_NODE], C=[])
    I_XZ = conditional_mutual_information_from_k(K, A=[X_NODE], B=[Z_NODE], C=[])
    return I_XY, I_XZ


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    here = Path(__file__).resolve().parent
    (here / "results").mkdir(parents=True, exist_ok=True)
    (here / "figures").mkdir(parents=True, exist_ok=True)

    # Fixed wiretap channels, entries ~ CN(0, 1).
    g = torch.Generator(device="cpu").manual_seed(SEED)
    H_Y = cn_randn(D, D, generator=g).to(DTYPE).to(DEVICE)
    H_Z = cn_randn(D_EVE, D, generator=g).to(DTYPE).to(DEVICE)

    # Uniform precoder F = sqrt(P/d) I_d: no secure shaping, at the budget.
    F = (math.sqrt(TOTAL_POWER / D) * torch.eye(D, dtype=DTYPE, device=DEVICE)).clone()
    F = F.requires_grad_(True)

    # PGA on U = I(X;Y) - I(X;Z) inside the Frobenius ball ||F||_F^2 <= P.
    IY_hist: list[float] = []
    IZ_hist: list[float] = []

    def closure() -> torch.Tensor:
        I_XY, I_XZ = wiretap_mi(F, H_Y, H_Z)
        IY_hist.append(I_XY.item())
        IZ_hist.append(I_XZ.item())
        return I_XY - I_XZ

    def projector(params: list[torch.Tensor]) -> list[torch.Tensor]:
        return [project_frobenius_ball(p, TOTAL_POWER) for p in params]

    U_history = pga_ascent(
        closure, [F],
        step_size=STEP_SIZE, num_iters=NUM_ITERS, projector=projector,
    )

    # Final post-update snapshot.
    with torch.no_grad():
        I_XY_f, I_XZ_f = wiretap_mi(F, H_Y, H_Z)
    IY_hist.append(I_XY_f.item())
    IZ_hist.append(I_XZ_f.item())

    snapshot_iters = np.arange(NUM_ITERS + 1, dtype=np.int64)
    IY_arr = np.array(IY_hist)
    IZ_arr = np.array(IZ_hist)
    secrecy = IY_arr - IZ_arr

    npz_path = here / "results" / "secure_precoding.npz"
    np.savez(
        npz_path,
        snapshot_iters=snapshot_iters,
        IY_traj=IY_arr, IZ_traj=IZ_arr,
        U_history=np.array(U_history, dtype=np.float64),
        H_Y=H_Y.detach().cpu().numpy(),
        H_Z=H_Z.detach().cpu().numpy(),
        F_star=F.detach().cpu().numpy(),
        config=dict(
            d=D, d_eve=D_EVE, total_power=TOTAL_POWER, noise_var=NOISE_VAR,
            num_iters=NUM_ITERS, step_size=STEP_SIZE, seed=SEED,
        ),
    )
    print(f"results -> {npz_path}")

    # Figure: I(X;Y) vs I(X;Z) (top) and the secrecy rate (bottom).
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    })

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.5, 4.7))
    axA.plot(snapshot_iters, IY_arr, "-", color="tab:blue", linewidth=1.0,
             label=r"$I(X;Y)$  (legitimate)")
    axA.plot(snapshot_iters, IZ_arr, "-", color="tab:red", linewidth=1.0,
             label=r"$I(X;Z)$  (eavesdropper)")
    axA.set_xlabel("projected-gradient iteration")
    axA.set_ylabel("mutual information (nats)")
    axA.set_title("(a) legitimate vs eavesdropper")
    axA.set_xlim(0, NUM_ITERS)
    axA.legend()

    axB.plot(snapshot_iters, secrecy, "-", color="0.4", linewidth=1.0)
    axB.set_xlabel("projected-gradient iteration")
    axB.set_ylabel(r"secrecy rate $I(X;Y) - I(X;Z)$ (nats)")
    axB.set_title("(b) secrecy rate")
    axB.set_xlim(0, NUM_ITERS)

    fig.tight_layout()
    fig_path = here / "figures" / "secure_precoding.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"figure  -> {fig_path}")


if __name__ == "__main__":
    main()
