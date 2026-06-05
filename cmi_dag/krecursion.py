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
`get_K`.

`hermitianize` and `get_K` are the same numerical primitives as in
`gaussian_dag.krecursion`; they are vendored here so this library is fully
self-contained (no `gaussian-dag` runtime dependency).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def hermitianize(A: torch.Tensor) -> torch.Tensor:
    """Symmetrize a square matrix by (A + A^H) / 2.

    This enforces exact Hermitian structure on tensors that should be Hermitian
    in theory but may drift due to floating-point round-off.
    """
    return 0.5 * (A + A.mH)


def get_K(
    K: dict[tuple[int, int], torch.Tensor],
    a: int,
    b: int,
) -> torch.Tensor:
    """Return K_{ab}, applying Hermitian flip when a < b.

    K is assumed to store only canonical keys (j, k) with j >= k.
    """
    if a >= b:
        return K[(a, b)]
    return K[(b, a)].mH


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


def compute_effective_channel(
    num_nodes: int,
    roots: Sequence[int],
    parents: dict[int, list[int]],
    edge_mats: dict[tuple[int, int], torch.Tensor],
    noise_covs: dict[int, torch.Tensor],
    *,
    source_dims: dict[int, int] | None = None,
    symmetrize_self_blocks: bool = True,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor]]:
    """Multi-root effective-channel representation (G, C) of a linear Gaussian DAG.

    Collapses the multi-root DAG to an equivalent multi-source linear Gaussian
    channel
        Y = sum_{r in roots} G_M^{(r)} X_r + R_M,   {X_r} mutually independent,
    exposing the per-root effective channel matrices G_j^{(r)} (gain from
    source root r to node j) and the effective-noise covariance blocks
    C_{jk} = E[R_j R_k^H].

    The per-root effective channel matrices follow, for each root r, the base
    case
        G_r^{(r)}  = I_{d_r},
        G_{r'}^{(r)} = 0           (other root r' != r; roots are independent),
    and the forward recursion over non-root nodes
        G_j^{(r)} = sum_{i in Pa(j)} A_{ji} G_i^{(r)},
    with G_j^{(r)} of shape (d_j, d_r). The effective-noise blocks obey the
    multi-root K-recursion with all root covariances set to zero; they are
    obtained here by reusing `compute_k_blocks_multiroot` with zero
    `root_covs`, which is exact because that recursion is affine in its
    root-covariance seeds. Together they satisfy the decomposition
        K_{jk} = sum_{r in roots} G_j^{(r)} Sigma_r G_k^{(r)H} + C_{jk}.

    This is the multi-root generalization of the single-root
    effective-channel representation (Remark "Effective-channel
    representation" of the gaussian-dag paper); with a single root
    (`roots=[0]`) it reduces to `gaussian_dag.compute_effective_channel`.
    The single-root case is published; the multi-source generalization here
    accompanies the multi-terminal / conditional-MI companion paper, which is
    in preparation.

    Args:
        num_nodes: Total number of nodes M (indices 0..M-1).
        roots: Indices of the root nodes. Must be exactly the prefix
            {0, ..., K-1} in topological order (same convention as
            `compute_k_blocks_multiroot`).
        parents: parents[j] = list of parent indices for non-root j. Must
            satisfy i < j for every i in parents[j].
        edge_mats: edge_mats[(j, i)] = A_{ji}, shape (d_j, d_i).
        noise_covs: noise_covs[j] = Sigma_j (shape d_j x d_j) for every
            non-root j. Note: the root covariances are intentionally NOT an
            argument; (G, C) describe the channel and are independent of them.
        source_dims: Optional dict {r: d_r} giving each root's dimension. If
            None, each d_r is inferred from any edge into root r
            (edge_mats[(j, r)] has shape (d_j, d_r)).
        symmetrize_self_blocks: If True, apply (A + A^H)/2 to each C self-cov
            block (forwarded to `compute_k_blocks_multiroot`).

    Returns:
        Tuple (G, C):
        - G: dict {(r, j): G_j^{(r)}} for r in roots and 0 <= j < num_nodes,
          each of shape (d_j, d_r). G[(r, r)] = I_{d_r} and G[(r, r')] = 0 for
          distinct roots r, r'.
        - C: dict of canonical blocks C[(j, k)] for 0 <= k <= j < num_nodes
          (same key convention as `compute_k_blocks_multiroot`; use `get_K`
          for the Hermitian flip), with all root self-blocks C[(r, r)] = 0.
        Both are differentiable in `edge_mats` (and C in `noise_covs`) via
        PyTorch autograd. Newly allocated tensors inherit dtype/device from
        the input tensors.

    Raises:
        ValueError: if `roots` is not the prefix {0, ..., K-1}, if a root's
            dimension cannot be inferred (no edge into it and no `source_dims`
            entry), if dtype/device cannot be inferred, if a provided
            `source_dims[r]` disagrees with the dimension implied by an edge
            into root r, or via the topological-order / missing-parent /
            missing-noise checks inherited from `compute_k_blocks_multiroot`.
    """
    roots = sorted(roots)
    num_roots = len(roots)
    if roots != list(range(num_roots)):
        raise ValueError(
            f"roots must be the prefix {{0, ..., K-1}} in topological order, "
            f"got {roots}."
        )

    # A reference tensor for dtype/device (any edge or noise matrix).
    ref = next(iter(edge_mats.values()), None)
    if ref is None:
        ref = next(iter(noise_covs.values()), None)
    if ref is None:
        raise ValueError(
            "Cannot infer dtype/device: edge_mats and noise_covs are both "
            "empty. Provide at least one edge or noise matrix."
        )

    # Resolve each root's dimension d_r (explicit override or edge inference).
    dims: dict[int, int] = {}
    for r in roots:
        edge_into_r = next(
            (edge_mats[(j, r)] for j in range(num_nodes) if (j, r) in edge_mats),
            None,
        )
        inferred = edge_into_r.shape[1] if edge_into_r is not None else None
        explicit = None if source_dims is None else source_dims.get(r)
        if explicit is not None and inferred is not None and explicit != inferred:
            raise ValueError(
                f"source_dims[{r}]={explicit} disagrees with the dimension "
                f"{inferred} implied by an edge into root {r}."
            )
        d_r = explicit if explicit is not None else inferred
        if d_r is None:
            raise ValueError(
                f"Cannot infer the dimension of root {r}: it has no outgoing "
                f"edge. Pass source_dims with an entry for root {r}."
            )
        dims[r] = d_r

    # Effective-noise blocks: multi-root K-recursion with zero root covariances.
    zero_root_covs = {
        r: torch.zeros(dims[r], dims[r], dtype=ref.dtype, device=ref.device)
        for r in roots
    }
    C = compute_k_blocks_multiroot(
        num_nodes,
        roots,
        parents,
        edge_mats,
        zero_root_covs,
        noise_covs,
        symmetrize_self_blocks=symmetrize_self_blocks,
    )

    # Per-root effective channel matrices via the forward gain recursion.
    G: dict[tuple[int, int], torch.Tensor] = {}
    for r in roots:
        # Base case over the root nodes.
        for r2 in roots:
            if r2 == r:
                G[(r, r2)] = torch.eye(dims[r], dtype=ref.dtype, device=ref.device)
            else:
                G[(r, r2)] = torch.zeros(
                    dims[r2], dims[r], dtype=ref.dtype, device=ref.device
                )
        # Non-root nodes, in topological order.
        for j in range(num_roots, num_nodes):
            acc: torch.Tensor | None = None
            for i in parents[j]:
                term = edge_mats[(j, i)] @ G[(r, i)]
                acc = term if acc is None else acc + term
            assert acc is not None  # parents[j] non-empty (validated by C above)
            G[(r, j)] = acc

    return G, C
