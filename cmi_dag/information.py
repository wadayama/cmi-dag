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
matrix inverse is formed (Cholesky factorization + `torch.cholesky_solve`
only); the log-dets use the Cholesky-based `logdet_hpd`. Every operation
is device-agnostic.

`logdet_hpd` is the same numerical primitive as in
`gaussian_dag.information`; it is vendored here so this library is fully
self-contained (no `gaussian-dag` runtime dependency).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from cmi_dag.krecursion import get_K, hermitianize


def logdet_hpd(A: torch.Tensor, jitter: float = 0.0) -> torch.Tensor:
    """Cholesky-based log-determinant for Hermitian positive-definite A.

    For Hermitian positive-definite A = L L^H with lower-triangular L and
    real positive diag(L),
        log det A = 2 * sum_i log L_ii.

    The input is symmetrized by (A + A^H)/2 to enforce Hermitian structure
    against floating-point drift before the Cholesky step. The Cholesky
    factorisation itself is performed via ``torch.linalg.cholesky_ex``: on
    failure (matrix not strictly positive-definite), this function raises
    a ``ValueError`` with diagnostic information rather than letting an
    opaque PyTorch error propagate out of the autograd graph.

    Args:
        A: Hermitian positive-definite matrix (shape d x d, complex or
            real). With the default ``jitter=0`` the input must be
            strictly positive-definite; otherwise the Cholesky factorisation
            fails and a ``ValueError`` is raised (see Raises). A
            rank-deficient or merely PSD input can be admitted by passing
            a small ``jitter > 0`` to regularise it back into the PD cone.
        jitter: If > 0, replace A by A + jitter * I before factorization.
            Useful when the underlying matrix is near-singular or rank
            deficient (e.g. because the noise covariance is small or
            structurally degenerate). Use sparingly: a non-zero jitter
            perturbs the log-determinant by order
            ``d * jitter / lambda_min(A)`` and should be reported in any
            experiment that uses it.

    Returns:
        Real scalar tensor: log det A (natural log; nats convention).

    Raises:
        ValueError: if A (after optional jitter) is not strictly positive
            definite. The error message indicates the leading minor where
            positive definiteness failed and suggests three remediations:
            (1) ensure that the terminal noise covariance is strictly
            positive definite so that the model-level regularity assumption
            holds; (2) pass a small ``jitter > 0`` to absorb round-off
            near-singularity; or (3) when called from inside a PGA/PGD
            loop, reduce the step size so that iterates stay inside the
            open positive-definite cone.
    """
    A = hermitianize(A)
    if jitter > 0.0:
        d = A.shape[-1]
        A = A + jitter * torch.eye(d, dtype=A.dtype, device=A.device)
    L, info = torch.linalg.cholesky_ex(A, check_errors=False)
    info_value = int(info.item())
    if info_value != 0:
        raise ValueError(
            "logdet_hpd: input matrix is not Hermitian positive definite "
            f"(Cholesky failed at leading minor of order {info_value}). "
            "Common remedies: (1) ensure the terminal noise covariance is "
            "strictly positive definite (the regularity assumption); "
            "(2) pass jitter>0 to logdet_hpd / "
            "conditional_mutual_information_from_k to absorb near-singularity; "
            "(3) inside pga_descent, reduce step_size so that iterates remain "
            "in the positive-definite cone."
        )
    return 2.0 * torch.log(torch.diagonal(L).real).sum()


def _assemble(
    K: dict[tuple[int, int], torch.Tensor],
    rows: Sequence[int],
    cols: Sequence[int],
) -> torch.Tensor:
    """Stack K-blocks into a single covariance Sigma_{rows, cols}.

    Block (r, c) is K_{rc} = E[V_r V_c^H], read via the Hermitian-flip
    helper `get_K`.

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
    *,
    jitter: float = 0.0,
) -> torch.Tensor:
    """Conditional covariance Sigma_{A|Z} via the Schur complement.

    Returns Sigma_{A,A} - Sigma_{A,Z} Sigma_{Z,Z}^{-1} Sigma_{Z,A} for
    non-empty Z, and the marginal Sigma_{A,A} when Z is empty.

    The inverse Sigma_{Z,Z}^{-1} is never formed: Sigma_{Z,Z} is Hermitian
    positive definite under the model regularity assumption, so it is
    Cholesky-factorized and the correction term is computed via
    `torch.cholesky_solve`. The column block Sigma_{Z,A} is obtained as
    Sigma_{A,Z}^H rather than re-assembled. If jitter > 0, Sigma_{Z,Z} is
    regularized to Sigma_{Z,Z} + jitter * I before factorization (the same
    remedy `logdet_hpd` applies to the conditional covariance itself).

    Raises:
        ValueError: if Sigma_{Z,Z} (after optional jitter) is not strictly
            Hermitian positive definite, with the same remediation hints
            as `logdet_hpd`.
    """
    Sigma_AA = _assemble(K, A, A)
    if len(Z) == 0:
        return Sigma_AA
    Sigma_AZ = _assemble(K, A, Z)
    Sigma_ZZ = hermitianize(_assemble(K, Z, Z))
    if jitter > 0.0:
        d = Sigma_ZZ.shape[-1]
        Sigma_ZZ = Sigma_ZZ + jitter * torch.eye(
            d, dtype=Sigma_ZZ.dtype, device=Sigma_ZZ.device
        )
    L, info = torch.linalg.cholesky_ex(Sigma_ZZ, check_errors=False)
    info_value = int(info.item())
    if info_value != 0:
        raise ValueError(
            f"Conditioning covariance Sigma_{{Z,Z}} for Z={list(Z)} is not "
            "Hermitian positive definite (Cholesky failed at leading minor "
            f"of order {info_value}). Common remedies: (1) ensure the noise "
            "covariances of the conditioning nodes are strictly positive "
            "definite (the regularity assumption); (2) pass jitter>0 to "
            "absorb near-singularity; (3) inside pga_ascent / pga_descent, "
            "reduce step_size so that iterates remain in the "
            "positive-definite cone."
        )
    return Sigma_AA - Sigma_AZ @ torch.cholesky_solve(Sigma_AZ.mH, L)


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

    Convention. Like the rest of the library, the log-det formula uses the
    circular complex Gaussian convention (no factor of 1/2; values in
    nats). For real-dtype inputs the same formula is applied unchanged,
    which is twice the mutual information of a real Gaussian vector; halve
    the result if the real-Gaussian convention is required.

    Args:
        K: Canonical K-blocks produced by either `compute_k_blocks` (parent,
            single-root) or `compute_k_blocks_multiroot` (this library,
            multi-root). Keys are (j, k) with j >= k.
        A: Node indices of the first information set (non-empty).
        B: Node indices of the second information set (non-empty).
        C: Conditioning node indices (default empty -> unconditional MI).
        jitter: Optional diagonal jitter applied (i) to the conditioning
            covariances Sigma_{C,C} and Sigma_{BC,BC} before their
            Cholesky-based Schur solves, and (ii) to Sigma_{A|C} and
            Sigma_{A|BC} inside `logdet_hpd`; useful when any of these
            covariances is nearly singular (rank-deficient controllable
            factors, low-SNR channels, etc.).

    Returns:
        Real scalar tensor in nats, on the same `dtype` / `device` as the
        K-blocks. Differentiable through K.

    Raises:
        ValueError: if A or B is empty, if A, B, C are not pairwise
            disjoint, or if a conditioning covariance Sigma_{C,C} /
            Sigma_{BC,BC} or a conditional covariance Sigma_{A|C} /
            Sigma_{A|BC} (after optional jitter) is not strictly Hermitian
            positive definite.
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

    Sigma_A_given_C = _conditional_cov(K, A, C, jitter=jitter)
    Sigma_A_given_BC = _conditional_cov(K, A, sorted(B + C), jitter=jitter)
    return logdet_hpd(Sigma_A_given_C, jitter=jitter) - logdet_hpd(
        Sigma_A_given_BC, jitter=jitter
    )


def conditional_differential_entropy_from_k(
    K: dict[tuple[int, int], torch.Tensor],
    A: Sequence[int],
    C: Sequence[int] = (),
    *,
    jitter: float = 0.0,
) -> torch.Tensor:
    """Conditional differential entropy h(V_A | V_C) from K-blocks (in nats).

    Implements the log-det closed form for a circular complex Gaussian,

        h(V_A | V_C) = log det(pi e * Sigma_{A|C})
                     = log det Sigma_{A|C} + d_A * log(pi e),

    where Sigma_{A|C} is the conditional covariance (Schur complement of the
    support block covariance) and d_A = sum_{a in A} dim(V_a) is the total
    (complex) dimension of the information set A. With an empty conditioning
    set C this reduces to the marginal differential entropy h(V_A).

    This is the "one-Schur, one-log-det" specialization of conditional MI:
    the conditional MI factors as a difference of conditional entropies,

        I(V_A; V_B | V_C) = h(V_A | V_C) - h(V_A | V_{BC}),

    so each `conditional_mutual_information_from_k` call is exactly two of
    these entropy evaluations. The same conditional-covariance and
    Cholesky-based log-det primitives are reused; only one Schur complement
    plus an additive constant is needed.

    Convention. Like the conditional MI, this uses the circular complex
    Gaussian convention (no factor of 1/2; values in nats). The additive
    constant d_A * log(pi e) is a pure constant in the design parameters,
    so it does not affect any gradient: only the log-det term carries
    autograd sensitivity. For real-dtype inputs the function still applies
    the complex convention, consistent with the rest of the library.

    Args:
        K: Canonical K-blocks produced by either `compute_k_blocks` (parent,
            single-root) or `compute_k_blocks_multiroot` (this library,
            multi-root). Keys are (j, k) with j >= k.
        A: Node indices of the information set (non-empty).
        C: Conditioning node indices (default empty -> marginal entropy).
        jitter: Optional diagonal jitter applied (i) to the conditioning
            covariance Sigma_{C,C} before its Cholesky-based Schur solve
            and (ii) to Sigma_{A|C} inside `logdet_hpd`; useful when
            either covariance is nearly singular (rank-deficient
            controllable factors, low-SNR channels, etc.).

    Returns:
        Real scalar tensor in nats, on the same `dtype` (real) / `device`
        as the K-blocks. Differentiable through K (via the log-det term).

    Raises:
        ValueError: if A is empty, if A and C are not disjoint, or if
            Sigma_{C,C} or Sigma_{A|C} (after optional jitter) is not
            strictly Hermitian positive definite.
    """
    A = sorted(A)
    C = sorted(C)
    if len(A) == 0:
        raise ValueError("A must be non-empty.")
    if set(A) & set(C):
        raise ValueError(f"A and C must be disjoint; got A={A}, C={C}.")

    Sigma_A_given_C = _conditional_cov(K, A, C, jitter=jitter)
    logdet = logdet_hpd(Sigma_A_given_C, jitter=jitter)
    d_A = sum(K[(a, a)].shape[-1] for a in A)
    return logdet + d_A * math.log(math.pi * math.e)
