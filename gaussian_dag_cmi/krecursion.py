"""Multi-root K-recursion for linear Gaussian DAGs.

Model (0-based indexing, multi-root extension of gaussian_dag.krecursion):
    Roots r in {0, ..., K-1}:  V_r ~ CN(0, Sigma_r), mutually independent.
    Non-roots j in {K, ..., M-1}:
        V_j = sum_{i in Pa(j)} A_{ji} V_i + Z_j,  Z_j ~ CN(0, Sigma_j),
    with all Z_j mutually independent and independent of the user inputs.

The single-root case (K=1) reduces to the parent's `compute_k_blocks`
(modulo the trivial relabeling input_cov = root_covs[0]). The multi-root
extension is required for multi-terminal channels (e.g., the 2-user MAC has
two transmitter roots).

Canonical storage: K stores only K_{jk} for j >= k. Access to K_{ab} with
a < b is performed via the Hermitian flip rule K_{ab} = K_{ba}^H through
the parent's `get_K`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from gaussian_dag.krecursion import get_K, hermitianize


def compute_k_blocks_multiroot(
    num_nodes: int,
    roots: Sequence[int],
    parents: dict[int, list[int]],
    edge_mats: dict[tuple[int, int], torch.Tensor],
    root_covs: dict[int, torch.Tensor],
    noise_covs: dict[int, torch.Tensor],
    *,
    symmetrize_self_blocks: bool = True,
) -> dict[tuple[int, int], torch.Tensor]:
    """Compute all canonical K-blocks K_{jk} for a multi-root linear Gaussian DAG.

    Multi-root analog of `gaussian_dag.compute_k_blocks`: the recursion
    starts from the per-root base case
        K_{rr}   = Sigma_r       (root input covariance, r in `roots`),
        K_{rr'}  = 0             (mutual independence of distinct roots),
    and then proceeds through the non-root nodes j in topological order:
        K_{jk}   = sum_{i in Pa(j)} A_{ji} K_{ik}                       (k < j)
        K_{jj}   = sum_{i,i' in Pa(j)} A_{ji} K_{ii'} A_{ji'}^H + Sigma_j.

    All newly allocated tensors (the zero off-diagonal root blocks) inherit
    `dtype` and `device` from `root_covs`, so the function is fully
    device-agnostic (CPU / CUDA / MPS).

    Args:
        num_nodes: Total number of nodes M (indices 0..M-1).
        roots: Indices of the root nodes. Must be exactly the prefix
            {0, ..., K-1} in topological order, with K = len(roots) and
            K < num_nodes (there must be at least one non-root node so the
            DAG is non-trivial as a channel).
        parents: parents[j] = list of parent indices for non-root j. Must
            satisfy i < j for every i in parents[j]. Roots need not appear
            as keys.
        edge_mats: edge_mats[(j, i)] = A_{ji}, the linear transformation on
            the edge i -> j (shape d_j x d_i).
        root_covs: root_covs[r] = Sigma_r (shape d_r x d_r) for r in roots,
            the input covariance of user / source r.
        noise_covs: noise_covs[j] = Sigma_j (shape d_j x d_j) for every
            non-root j.
        symmetrize_self_blocks: If True, apply (A + A^H)/2 to each self-cov
            block K_{jj} (including the per-root K_{rr}) to enforce
            Hermitian structure numerically.

    Returns:
        Dictionary K with keys (j, k) for 0 <= k <= j < num_nodes. Every
        block is differentiable through `edge_mats`, `root_covs`, and
        `noise_covs` via PyTorch autograd.

    Raises:
        ValueError: if `roots` is not the prefix {0, ..., K-1} in
            topological order, if K >= num_nodes (no non-root node), if a
            non-root has empty / out-of-order parents, or if a non-root is
            missing from `noise_covs`.
    """
    roots = sorted(roots)
    num_roots = len(roots)
    if roots != list(range(num_roots)):
        raise ValueError(
            f"roots must be the prefix {{0, ..., K-1}} in topological order, "
            f"got {roots}."
        )
    if num_roots >= num_nodes:
        raise ValueError(
            f"num_roots ({num_roots}) must be strictly less than num_nodes "
            f"({num_nodes}): the DAG must contain at least one non-root node "
            "for the channel to be non-trivial."
        )
    for r in roots:
        if r not in root_covs:
            raise ValueError(f"root_covs is missing the entry for root {r}.")

    K: dict[tuple[int, int], torch.Tensor] = {}

    # Base case: root self-covariances and zero cross-covariances.
    for r in roots:
        cov = root_covs[r]
        K[(r, r)] = hermitianize(cov) if symmetrize_self_blocks else cov
    for r in roots:
        for r2 in roots:
            if r2 < r:
                d_r = K[(r, r)].shape[-1]
                d_r2 = K[(r2, r2)].shape[-1]
                K[(r, r2)] = torch.zeros(
                    d_r, d_r2, dtype=K[(r, r)].dtype, device=K[(r, r)].device
                )

    # Non-root nodes, in topological order.
    for j in range(num_roots, num_nodes):
        if j not in parents or len(parents[j]) == 0:
            raise ValueError(f"Non-root node {j} has no parents.")
        for i in parents[j]:
            if not (0 <= i < j):
                raise ValueError(
                    f"Parent {i} of node {j} violates topological order "
                    f"(0 <= i < j)."
                )
        if j not in noise_covs:
            raise ValueError(f"noise_covs is missing the entry for non-root node {j}.")

        # (1) Cross blocks K_{jk} for k = 0, ..., j-1.
        for k in range(j):
            acc: torch.Tensor | None = None
            for i in parents[j]:
                term = edge_mats[(j, i)] @ get_K(K, i, k)
                acc = term if acc is None else acc + term
            assert acc is not None  # parents[j] is non-empty
            K[(j, k)] = acc

        # (2) Self block K_{jj} = sum_{i,i'} A_{ji} K_{ii'} A_{ji'}^H + Sigma_j.
        acc = noise_covs[j]
        for i in parents[j]:
            Aji = edge_mats[(j, i)]
            for ip in parents[j]:
                Ajip = edge_mats[(j, ip)]
                acc = acc + Aji @ get_K(K, i, ip) @ Ajip.mH
        K[(j, j)] = hermitianize(acc) if symmetrize_self_blocks else acc

    return K
