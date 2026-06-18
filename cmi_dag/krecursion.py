"""Multi-root K-recursion for linear Gaussian DAGs.

Model (0-based indexing, multi-root extension of gaussian_dag.krecursion):
    Roots r in {0, ..., K-1}:  V_r ~ CN(0, Sigma_r),
        mutually independent by default; optional cross-covariances
        Sigma_{r,r'} = E[V_r V_{r'}^H] may be supplied to model correlated
        sources (relevant to multi-terminal source compression, Slepian-Wolf
        / CEO problems, common-information).
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


def _assemble_root_block(
    roots: Sequence[int],
    root_covs: dict[int, torch.Tensor],
    cross_root_covs: dict[tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    """Assemble the joint root covariance matrix Sigma_{R, R}.

    Stack the per-root self-blocks Sigma_r along the diagonal and place the
    user-supplied cross-blocks Sigma_{r, r'} (key convention r > r') in the
    lower triangle, with the Hermitian transpose Sigma_{r,r'}^H in the upper
    triangle. Missing cross keys default to zero.

    The returned tensor is used only for the joint-PD validation Cholesky
    check; its autograd graph (if any) is discarded by the caller.

    Args:
        roots: Sorted prefix {0, ..., K-1}.
        root_covs: root_covs[r] = Sigma_r (shape d_r x d_r) for r in roots.
        cross_root_covs: cross_root_covs[(r, r')] = Sigma_{r, r'}
            (shape d_r x d_{r'}) for canonical pairs r > r'.

    Returns:
        Hermitian matrix of shape (sum_r d_r, sum_r d_r) on the same dtype
        and device as the inputs.
    """
    row_strips = []
    for r in roots:
        blocks = []
        for r2 in roots:
            if r == r2:
                blocks.append(root_covs[r])
            elif r > r2:
                sigma = cross_root_covs.get((r, r2))
                if sigma is None:
                    blocks.append(
                        torch.zeros(
                            root_covs[r].shape[-1],
                            root_covs[r2].shape[-1],
                            dtype=root_covs[r].dtype,
                            device=root_covs[r].device,
                        )
                    )
                else:
                    blocks.append(sigma)
            else:  # r < r2: Hermitian transpose of the lower-triangular block
                sigma = cross_root_covs.get((r2, r))
                if sigma is None:
                    blocks.append(
                        torch.zeros(
                            root_covs[r].shape[-1],
                            root_covs[r2].shape[-1],
                            dtype=root_covs[r].dtype,
                            device=root_covs[r].device,
                        )
                    )
                else:
                    blocks.append(sigma.mH)
        row_strips.append(blocks)
    # Broadcast every block to the common leading batch shape before stacking,
    # so batched root covariances coexist with unbatched zero cross-blocks.
    flat = [b for strip in row_strips for b in strip]
    batch = torch.broadcast_shapes(*(b.shape[:-2] for b in flat))

    def _b(block: torch.Tensor) -> torch.Tensor:
        return block.expand(*batch, block.shape[-2], block.shape[-1])

    return torch.cat(
        [torch.cat([_b(b) for b in strip], dim=-1) for strip in row_strips],
        dim=-2,
    )


def _validate_cross_root_covs(
    roots: Sequence[int],
    root_covs: dict[int, torch.Tensor],
    cross_root_covs: dict[tuple[int, int], torch.Tensor],
) -> None:
    """Validate the cross-root covariance dict and check joint PD-ness.

    Performs (in order):
      1. Key-shape checks: each key is (r, r') with r > r' and both in roots.
         Self-keys (r, r) are rejected with a hint to use root_covs.
         Reversed keys (r', r) with r' > r are rejected with a hint that the
         canonical convention is lower-triangular (r > r').
      2. Tensor-shape checks: cross_root_covs[(r, r')] has shape
         (d_r, d_{r'}) consistent with root_covs[r] and root_covs[r'].
      3. Joint Hermitian-PD check on the assembled Sigma_{R, R} via
         torch.linalg.cholesky_ex. Performed outside autograd by detaching
         the assembled matrix (only the info code is consumed).

    Args:
        roots: Sorted prefix {0, ..., K-1}.
        root_covs: Per-root self-covariances.
        cross_root_covs: User-supplied cross blocks (may be empty).

    Raises:
        ValueError: on any of the above failures, with a diagnostic message.
    """
    roots_set = set(roots)
    for key in cross_root_covs:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError(
                f"cross_root_covs key must be a 2-tuple of ints, got {key!r}."
            )
        r, r2 = key
        if r == r2:
            raise ValueError(
                f"cross_root_covs key ({r}, {r2}) is a self-pair; "
                "self-covariances must be supplied via root_covs[r] instead."
            )
        if r < r2:
            raise ValueError(
                f"cross_root_covs key ({r}, {r2}) violates the canonical "
                "lower-triangular convention (r > r'). Provide Sigma_{r, r'} "
                f"under the key ({r2}, {r}) instead; the upper triangle is "
                "obtained internally via the Hermitian flip."
            )
        if r not in roots_set or r2 not in roots_set:
            raise ValueError(
                f"cross_root_covs key ({r}, {r2}) refers to a non-root index; "
                f"roots are {list(roots)}."
            )
        sigma = cross_root_covs[key]
        d_r = root_covs[r].shape[-1]
        d_r2 = root_covs[r2].shape[-1]
        if sigma.shape[-2:] != (d_r, d_r2):
            raise ValueError(
                f"cross_root_covs[({r}, {r2})] has shape {tuple(sigma.shape)}, "
                f"expected ({d_r}, {d_r2}) to match root_covs."
            )
        if sigma.dtype != root_covs[r].dtype:
            raise ValueError(
                f"cross_root_covs[({r}, {r2})] dtype {sigma.dtype} does not "
                f"match root_covs[{r}] dtype {root_covs[r].dtype}."
            )
        if sigma.device != root_covs[r].device:
            raise ValueError(
                f"cross_root_covs[({r}, {r2})] device {sigma.device} does not "
                f"match root_covs[{r}] device {root_covs[r].device}."
            )

    # Joint PD check (assembled Sigma_{R, R}). Detach so the Cholesky check
    # cannot leak into the autograd graph of downstream K-blocks.
    Sigma_RR = _assemble_root_block(roots, root_covs, cross_root_covs).detach()
    Sigma_RR = 0.5 * (Sigma_RR + Sigma_RR.mH)
    _, info = torch.linalg.cholesky_ex(Sigma_RR, check_errors=False)
    if torch.any(info != 0):
        flat = info.reshape(-1)
        first = int(torch.nonzero(flat != 0, as_tuple=False)[0])
        info_value = int(flat[first])
        n_fail, total = int((flat != 0).sum()), int(flat.numel())
        # Heuristic: map the leading-minor index back to a root pair via
        # cumulative dimensions (for the first failing batch element).
        cum = 0
        which_root = roots[-1]
        for r in roots:
            d_r = root_covs[r].shape[-1]
            if cum + d_r >= info_value:
                which_root = r
                break
            cum += d_r
        where = ("" if total == 1
                 else f" ({n_fail} of {total} batch elements; first at index "
                      f"{first})")
        raise ValueError(
            "Joint root covariance Sigma_{R, R} (assembled from root_covs "
            "and cross_root_covs) is not Hermitian positive definite: "
            f"Cholesky failed at leading minor of order {info_value} "
            f"(within the block of root {which_root}){where}. Common remedies: "
            "(1) reduce the magnitude of cross_root_covs entries; "
            "(2) inflate root_covs diagonals; (3) verify that "
            "cross_root_covs keys follow the canonical (r > r') convention."
        )


def compute_k_blocks_multiroot(
    num_nodes: int,
    roots: Sequence[int],
    parents: dict[int, list[int]],
    edge_mats: dict[tuple[int, int], torch.Tensor],
    root_covs: dict[int, torch.Tensor],
    noise_covs: dict[int, torch.Tensor],
    *,
    cross_root_covs: dict[tuple[int, int], torch.Tensor] | None = None,
    symmetrize_self_blocks: bool = True,
) -> dict[tuple[int, int], torch.Tensor]:
    """Compute all canonical K-blocks K_{jk} for a multi-root linear Gaussian DAG.

    Multi-root analog of `gaussian_dag.compute_k_blocks`: the recursion
    starts from the per-root base case
        K_{rr}   = Sigma_r              (root input covariance, r in `roots`),
        K_{rr'}  = Sigma_{r, r'}        (cross covariance, r > r'; default 0),
    and then proceeds through the non-root nodes j in topological order:
        K_{jk}   = sum_{i in Pa(j)} A_{ji} K_{ik}                       (k < j)
        K_{jj}   = sum_{i,i' in Pa(j)} A_{ji} K_{ii'} A_{ji'}^H + Sigma_j.

    By default the roots are mutually independent (Sigma_{r, r'} = 0). The
    optional `cross_root_covs` argument seeds non-zero cross covariances —
    useful for multi-terminal source compression (Slepian-Wolf / CEO /
    common-information settings). The forward recursion (2.5) is unchanged
    because it is affine in the base seed; the independent and correlated
    cases share a single implementation path.

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
        cross_root_covs: Optional dict {(r, r'): Sigma_{r, r'}} encoding
            cross-covariances E[V_r V_{r'}^H] of distinct roots. Keys must
            satisfy r > r' (canonical lower-triangular convention; the upper
            triangle Sigma_{r', r} = Sigma_{r, r'}^H is reconstructed
            internally). Missing keys default to zero (independent roots).
            When non-empty, the assembled joint root covariance
            Sigma_{R, R} is checked to be Hermitian positive definite by a
            Cholesky factorisation performed outside the autograd tape;
            failure raises ValueError. If `None` or `{}` (the default),
            both validation and the new code path are skipped and the
            behaviour is byte-identical to the independent-roots case.
        symmetrize_self_blocks: If True, apply (A + A^H)/2 to each self-cov
            block K_{jj} (including the per-root K_{rr}) to enforce
            Hermitian structure numerically. Cross blocks K_{r, r'} for
            r != r' are not symmetrized (they are off-diagonal sub-blocks of
            the joint root covariance, not self-blocks).

    Returns:
        Dictionary K with keys (j, k) for 0 <= k <= j < num_nodes. Every
        block is differentiable through `edge_mats`, `root_covs`,
        `noise_covs`, and (when supplied) `cross_root_covs` via PyTorch
        autograd.

    Raises:
        ValueError: if `roots` is not the prefix {0, ..., K-1} in
            topological order, if K >= num_nodes (no non-root node), if a
            non-root has empty / duplicate / out-of-order parents, if an
            edge declared in `parents` is missing from `edge_mats`, if a
            non-root is missing from `noise_covs`, if `cross_root_covs` violates the
            key-shape / tensor-shape / dtype / device contract, or if the
            assembled joint root covariance is not Hermitian positive
            definite.
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

    if cross_root_covs is not None and len(cross_root_covs) > 0:
        _validate_cross_root_covs(roots, root_covs, cross_root_covs)
    else:
        cross_root_covs = {}

    K: dict[tuple[int, int], torch.Tensor] = {}

    # Base case: root self-covariances and (optionally non-zero) cross-covariances.
    for r in roots:
        cov = root_covs[r]
        K[(r, r)] = hermitianize(cov) if symmetrize_self_blocks else cov
    for r in roots:
        for r2 in roots:
            if r2 < r:
                sigma = cross_root_covs.get((r, r2))
                if sigma is None:
                    d_r = K[(r, r)].shape[-1]
                    d_r2 = K[(r2, r2)].shape[-1]
                    K[(r, r2)] = torch.zeros(
                        d_r, d_r2, dtype=K[(r, r)].dtype, device=K[(r, r)].device
                    )
                else:
                    K[(r, r2)] = sigma

    # Non-root nodes, in topological order.
    for j in range(num_roots, num_nodes):
        if j not in parents or len(parents[j]) == 0:
            raise ValueError(f"Non-root node {j} has no parents.")
        if len(set(parents[j])) != len(parents[j]):
            raise ValueError(
                f"parents[{j}] contains duplicate entries: {parents[j]}. "
                "Each parent must appear exactly once (a duplicate would "
                "silently double-count its edge contribution)."
            )
        for i in parents[j]:
            if not (0 <= i < j):
                raise ValueError(
                    f"Parent {i} of node {j} violates topological order "
                    f"(0 <= i < j)."
                )
            if (j, i) not in edge_mats:
                raise ValueError(
                    f"edge_mats is missing the entry for edge ({j}, {i}) "
                    f"declared in parents[{j}]."
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
        Y = sum_{r in roots} G_M^{(r)} X_r + R_M,
    exposing the per-root effective channel matrices G_j^{(r)} (gain from
    source root r to node j) and the effective-noise covariance blocks
    C_{jk} = E[R_j R_k^H].

    The per-root effective channel matrices follow, for each root r, the base
    case
        G_r^{(r)}  = I_{d_r},
        G_{r'}^{(r)} = 0           (other root r' != r),
    and the forward recursion over non-root nodes
        G_j^{(r)} = sum_{i in Pa(j)} A_{ji} G_i^{(r)},
    with G_j^{(r)} of shape (d_j, d_r). The effective-noise blocks obey the
    multi-root K-recursion with all root covariances set to zero; they are
    obtained here by reusing `compute_k_blocks_multiroot` with zero
    `root_covs`, which is exact because that recursion is affine in its
    root-covariance seeds. The (G, C) representation is **intrinsic to the
    channel** and does not depend on the source distribution.

    Decomposition. Under independent sources X_r with covariance Sigma_r,
        K_{jk} = sum_{r in roots} G_j^{(r)} Sigma_r G_k^{(r)H} + C_{jk}.
    Under correlated sources (cross_root_covs supplied to
    `compute_k_blocks_multiroot`) the decomposition generalises to
        K_{jk} = sum_{r, r' in roots} G_j^{(r)} Sigma_{r, r'} G_k^{(r')H}
                  + C_{jk},
    with Sigma_{r, r} = root_covs[r] and Sigma_{r, r'} = cross_root_covs[(r, r')]
    (or its Hermitian transpose). The same (G, C) returned by this function
    is the correct decomposition in both cases; only the source-covariance
    block matrix changes.

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
