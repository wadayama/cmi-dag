"""Rate-function evaluation for multi-terminal rate regions.

A multi-terminal rate region is described by a finite intersection of
log-determinant inequalities

    R(eta, H) = { R in R_+^K : sum_{k in T} R_k <= f_T(eta, H), forall T in S },

where each rate function f_T is a finite real-linear combination of
conditional mutual informations

    f_T = sum_{n=1}^{N_T} alpha_{T,n} I(V_{A_n}; V_{B_n} | V_{C_n}),

with channel-specific disjoint subsets (A_n, B_n, C_n) (manuscript Sec. III).
This module provides:

- `Summand`, the type of one (alpha, A, B, C) line in the sum.
- `evaluate_rate_functions`, which evaluates {f_T}_{T in S} given the
  K-blocks of a forward K-recursion pass.

The MAC pentagon is the special case |S| = 3 with N_T = 1 for every T,
and alpha = 1; the Han-Kobayashi and DF/CF inner bounds typically have
N_T >= 2 and may include negative alpha (differences of CMIs).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from gaussian_dag_cmi.information import conditional_mutual_information_from_k

# One summand alpha_{T,n} I(V_{A_n}; V_{B_n} | V_{C_n}) of the rate function f_T.
Summand = tuple[float, Sequence[int], Sequence[int], Sequence[int]]


def evaluate_rate_functions(
    K: dict[tuple[int, int], torch.Tensor],
    inequalities: Sequence[Sequence[Summand]],
    *,
    jitter: float = 0.0,
) -> list[torch.Tensor]:
    """Evaluate the rate functions {f_T} for a general multi-terminal region.

    Args:
        K: Canonical K-blocks produced by `compute_k_blocks_multiroot` (or
            the parent's `compute_k_blocks` in the single-root special case).
        inequalities: Sequence of rate functions; each rate function is a
            sequence of `Summand` tuples (alpha, A, B, C) and evaluates to

                f_T = sum_n alpha_n * I(V_{A_n}; V_{B_n} | V_{C_n}).

            No constraint on the sign of `alpha`: the framework needs only
            differentiability of f_T in the design parameter, not
            concavity or monotonicity.
        jitter: Optional diagonal jitter forwarded to every conditional MI
            evaluation (see `conditional_mutual_information_from_k`).

    Returns:
        List of length `len(inequalities)`. Each entry is a real scalar
        tensor (nats) on the same `dtype` / `device` as the K-blocks,
        differentiable through K.

    Raises:
        ValueError: if any rate function has zero summands.
    """
    out: list[torch.Tensor] = []
    for summands in inequalities:
        f_T: torch.Tensor | None = None
        for alpha, A, B, C in summands:
            term = alpha * conditional_mutual_information_from_k(
                K, A, B, C, jitter=jitter
            )
            f_T = term if f_T is None else f_T + term
        if f_T is None:
            raise ValueError("Each rate function must have at least one summand.")
        out.append(f_T)
    return out
