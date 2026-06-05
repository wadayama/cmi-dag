"""Independent single-root reference oracle for cross-validating cmi-dag.

This module vendors the *single-root* K-recursion and single-pair mutual
information from `gaussian_dag` (the `gaussian-dag` library). They are kept
here, in the test tree only, purely as an independent reference against which
the multi-root implementation in `cmi_dag` is checked:

- `compute_k_blocks` here is gaussian-dag's dedicated single-root recursion.
  `cmi_dag.compute_k_blocks_multiroot` with `roots=[0]` must reproduce it
  block-for-block (see `test_single_root_matches_parent`).
- `mutual_information_from_k` here is gaussian-dag's single-pair MI.
  `cmi_dag.conditional_mutual_information_from_k(A=[x], B=[y], C=[])` must
  reproduce it (see `test_unconditional_matches_parent_single_link`).

These are reference oracles, NOT part of cmi-dag's public API. cmi-dag's own
multi-root recursion subsumes the single-root case; this file exists only so
the test suite can cross-check against the structurally distinct single-root
code that gaussian-dag derived independently.

The low-level accessors (`get_K`, `hermitianize`, `logdet_hpd`) are imported
from `cmi_dag` because they are byte-identical to gaussian-dag's; the
independence of the cross-check lives in the recursion / Schur-complement
structure below, exactly as it did when this oracle was gaussian-dag itself.
"""

from __future__ import annotations

import torch

from cmi_dag.information import logdet_hpd
from cmi_dag.krecursion import get_K, hermitianize


def compute_k_blocks(
    num_nodes: int,
    parents: dict[int, list[int]],
    edge_mats: dict[tuple[int, int], torch.Tensor],
    input_cov: torch.Tensor,
    noise_covs: dict[int, torch.Tensor],
    *,
    symmetrize_self_blocks: bool = True,
) -> dict[tuple[int, int], torch.Tensor]:
    """Single-root K-recursion (gaussian-dag reference). Node 0 is the unique root."""
    K: dict[tuple[int, int], torch.Tensor] = {}
    K[(0, 0)] = hermitianize(input_cov) if symmetrize_self_blocks else input_cov

    for j in range(1, num_nodes):
        if j not in parents or len(parents[j]) == 0:
            raise ValueError(f"Non-root node {j} has no parents.")
        for i in parents[j]:
            if i >= j:
                raise ValueError(
                    f"Parent {i} of node {j} violates topological order (i < j)."
                )

        # (1) Cross blocks K_{jk} for k = 0, ..., j-1.
        for k in range(j):
            acc = None
            for i in parents[j]:
                term = edge_mats[(j, i)] @ get_K(K, i, k)
                acc = term if acc is None else acc + term
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


def mutual_information_from_k(
    K: dict[tuple[int, int], torch.Tensor],
    output_node: int,
    input_node: int = 0,
    *,
    jitter: float = 0.0,
) -> torch.Tensor:
    """Single-pair I(X; Y) = log det Sigma_Y - log det Sigma_{Y|X} (gaussian-dag reference)."""
    K_yy = get_K(K, output_node, output_node)
    K_yx = get_K(K, output_node, input_node)
    K_xx = get_K(K, input_node, input_node)

    # Schur complement: Sigma_{Y|X} = Kyy - Kyx * Kxx^{-1} * Kxy.
    Kxx_inv_Kxy = torch.linalg.solve(K_xx, K_yx.mH)
    K_y_given_x = K_yy - K_yx @ Kxx_inv_Kxy

    return logdet_hpd(K_yy, jitter=jitter) - logdet_hpd(K_y_given_x, jitter=jitter)
