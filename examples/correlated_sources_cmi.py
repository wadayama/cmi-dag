"""Conditional MI between correlated sources observed through a noisy channel.

Two scalar transmitters X_1, X_2 are jointly complex-Gaussian with unit
variance and correlation coefficient rho; a single receiver observes
    Y = X_1 + X_2 + Z,   Z ~ CN(0, sigma_z^2).
We sweep rho over [-0.95, 0.95] and compute both
    I(X_1; X_2)        — native source correlation,
    I(X_1; X_2 | Y)    — residual correlation given the channel output,
illustrating the multi-terminal-source-compression setting that motivates
the optional `cross_root_covs` argument to `compute_k_blocks_multiroot`.

Usage:
    uv sync --extra examples
    uv run python examples/correlated_sources_cmi.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cmi_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
)

DTYPE = torch.complex128
SIGMA_Z = 0.5
RHOS = np.linspace(-0.95, 0.95, 39)


def cmi_pair(rho: float) -> tuple[float, float]:
    """Return (I(X_1; X_2), I(X_1; X_2 | Y)) at correlation coefficient rho."""
    cross = {(1, 0): torch.tensor([[rho + 0.0j]], dtype=DTYPE)}
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): torch.eye(1, dtype=DTYPE),
                   (2, 1): torch.eye(1, dtype=DTYPE)},
        root_covs={0: torch.eye(1, dtype=DTYPE), 1: torch.eye(1, dtype=DTYPE)},
        noise_covs={2: (SIGMA_Z ** 2) * torch.eye(1, dtype=DTYPE)},
        cross_root_covs=cross,
    )
    I_native = conditional_mutual_information_from_k(K, A=[0], B=[1]).item()
    I_given_Y = conditional_mutual_information_from_k(K, A=[0], B=[1], C=[2]).item()
    return I_native, I_given_Y


def main() -> None:
    rows = np.array([cmi_pair(float(r)) for r in RHOS])
    I_native, I_given_Y = rows[:, 0], rows[:, 1]

    print(f"{'rho':>8s}  {'I(X1;X2)':>12s}  {'I(X1;X2|Y)':>12s}")
    for r, a, b in zip(RHOS, I_native, I_given_Y):
        print(f"{r:>8.3f}  {a:>12.4f}  {b:>12.4f}")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    np.savez(results_dir / "correlated_sources_cmi.npz",
             rho=RHOS, I_native=I_native, I_given_Y=I_given_Y)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping plot. "
              "Install with `uv sync --extra examples`.)")
        return

    figs_dir = Path(__file__).parent / "figures"
    figs_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(RHOS, I_native, label=r"$I(X_1; X_2)$", lw=2)
    ax.plot(RHOS, I_given_Y, label=r"$I(X_1; X_2 \mid Y)$", lw=2, linestyle="--")
    ax.set_xlabel(r"source correlation $\rho$")
    ax.set_ylabel("mutual information (nats)")
    ax.set_title(r"Correlated sources $X_1, X_2$ through $Y = X_1 + X_2 + Z$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs_dir / "correlated_sources_cmi.pdf")
    fig.savefig(figs_dir / "correlated_sources_cmi.png", dpi=144)
    print(f"\nWrote figure to {figs_dir / 'correlated_sources_cmi.png'}")


if __name__ == "__main__":
    main()
