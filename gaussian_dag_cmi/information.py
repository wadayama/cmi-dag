"""Conditional mutual information I(V_A; V_B | V_C) on arbitrary disjoint subsets.

Extension of `gaussian_dag.mutual_information_from_k` (single-pair MI between
one input and one output node, given the input as the conditioning set) to
the multi-terminal setting, where A, B, C are arbitrary disjoint subsets of
the DAG nodes.

The conditional MI follows from the log-det formula

    I(V_A; V_B | V_C) = log det Sigma_{A|C} - log det Sigma_{A|BC},

where Sigma_{A|Z} is the conditional covariance, obtained as the Schur
complement

    Sigma_{A|Z} = Sigma_{A,A} - Sigma_{A,Z} Sigma_{Z,Z}^{-1} Sigma_{Z,A}.

All sub-block covariances are read from the canonical K-blocks
{K_{jk}} of the (multi-root) K-recursion by block extraction; no explicit
matrix inverse is formed (linear solve only); the log-dets use the parent's
Cholesky-based `logdet_hpd`. Every operation is device-agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from gaussian_dag.information import logdet_hpd
from gaussian_dag.krecursion import get_K


def _assemble(
    K: dict[tuple[int, int], torch.Tensor],
    rows: Sequence[int],
    cols: Sequence[int],
) -> torch.Tensor:
    """Stack K-blocks into a single covariance Sigma_{rows, cols}.

    Block (r, c) is K_{rc} = E[V_r V_c^H], read via the Hermitian-flip
    helper `get_K` (which inherits from the parent library).

    Returns a tensor of shape (sum_{r in rows} d_r, sum_{c in cols} d_c)
    on the same `dtype` / `device` as the underlying K-blocks.
    """
    row_strips = []
    for r in rows:
        blocks = [get_K(K, r, c) for c in cols]
        row_strips.append(torch.cat(blocks, dim=-1))
    return torch.cat(row_strips, dim=-2)


def _conditional_cov(
    K: dict[tuple[int, int], torch.Tensor],
    A: Sequence[int],
    Z: Sequence[int],
) -> torch.Tensor:
    """Conditional covariance Sigma_{A|Z} via the Schur complement.

    Returns Sigma_{A,A} - Sigma_{A,Z} Sigma_{Z,Z}^{-1} Sigma_{Z,A} for
    non-empty Z, and the marginal Sigma_{A,A} when Z is empty.

    The inverse Sigma_{Z,Z}^{-1} is never formed: the correction term is
    computed via a single `torch.linalg.solve`.
    """
    Sigma_AA = _assemble(K, A, A)
    if len(Z) == 0:
        return Sigma_AA
    Sigma_AZ = _assemble(K, A, Z)
    Sigma_ZZ = _assemble(K, Z, Z)
    Sigma_ZA = _assemble(K, Z, A)
    return Sigma_AA - Sigma_AZ @ torch.linalg.solve(Sigma_ZZ, Sigma_ZA)


def conditional_mutual_information_from_k(
    K: dict[tuple[int, int], torch.Tensor],
    A: Sequence[int],
    B: Sequence[int],
    C: Sequence[int] = (),
    *,
    jitter: float = 0.0,
) -> torch.Tensor:
    """Conditional mutual information I(V_A; V_B | V_C) from K-blocks.

    Implements the log-det closed form

        I(V_A; V_B | V_C) = log det Sigma_{A|C} - log det Sigma_{A|BC}

    for arbitrary disjoint subsets A, B, C of the DAG nodes. With an empty
    conditioning set C this reduces to the unconditional MI I(V_A; V_B),
    which agrees with the parent's `mutual_information_from_k` when A and
    B are singletons.

    Args:
        K: Canonical K-blocks produced by either `compute_k_blocks` (parent,
            single-root) or `compute_k_blocks_multiroot` (this library,
            multi-root). Keys are (j, k) with j >= k.
        A: Node indices of the first information set (non-empty).
        B: Node indices of the second information set (non-empty).
        C: Conditioning node indices (default empty -> unconditional MI).
        jitter: Optional diagonal jitter passed to `logdet_hpd` for both
            Sigma_{A|C} and Sigma_{A|BC}; useful when these conditional
            covariances are nearly singular (rank-deficient controllable
            factors, low-SNR channels, etc.).

    Returns:
        Real scalar tensor in nats, on the same `dtype` / `device` as the
        K-blocks. Differentiable through K.

    Raises:
        ValueError: if A or B is empty, or if A, B, C are not pairwise
            disjoint.
    """
    A = sorted(A)
    B = sorted(B)
    C = sorted(C)
    if len(A) == 0 or len(B) == 0:
        raise ValueError("A and B must both be non-empty.")
    all_nodes = A + B + C
    if len(all_nodes) != len(set(all_nodes)):
        raise ValueError(
            f"A, B, C must be pairwise disjoint; got A={A}, B={B}, C={C}."
        )

    Sigma_A_given_C = _conditional_cov(K, A, C)
    Sigma_A_given_BC = _conditional_cov(K, A, sorted(B + C))
    return logdet_hpd(Sigma_A_given_C, jitter=jitter) - logdet_hpd(
        Sigma_A_given_BC, jitter=jitter
    )
