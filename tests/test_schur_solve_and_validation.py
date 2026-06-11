"""Tests for the Cholesky-based Schur solve (jitter / diagnostics on the
conditioning block) and the structural input validation of
`compute_k_blocks_multiroot` (duplicate parents, missing edge matrices)."""

from __future__ import annotations

import pytest
import torch

from cmi_dag.information import conditional_mutual_information_from_k
from cmi_dag.krecursion import compute_k_blocks_multiroot, get_K

DTYPE = torch.complex128


def _randn_complex(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(*shape, dtype=torch.float64, generator=g)
    imag = torch.randn(*shape, dtype=torch.float64, generator=g)
    return torch.complex(real, imag)


def _build_mac_K(d: int, *, seed: int):
    A0 = _randn_complex(d, d, seed=seed)
    A1 = _randn_complex(d, d, seed=seed + 1)
    return compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): A0, (2, 1): A1},
        root_covs={0: torch.eye(d, dtype=DTYPE),
                   1: torch.eye(d, dtype=DTYPE)},
        noise_covs={2: torch.eye(d, dtype=DTYPE)},
    )


def _build_duplicated_node_K(d: int, *, seed: int):
    """4-node DAG where node 1 is a noiseless identity copy of root 0, so
    any conditioning set containing both {0, 1} has a singular covariance.
    Nodes 2 and 3 observe root 0 through independent noisy channels."""
    H2 = _randn_complex(d, d, seed=seed)
    H3 = _randn_complex(d, d, seed=seed + 1)
    return compute_k_blocks_multiroot(
        num_nodes=4, roots=[0],
        parents={1: [0], 2: [0], 3: [0]},
        edge_mats={(1, 0): torch.eye(d, dtype=DTYPE),
                   (2, 0): H2, (3, 0): H3},
        root_covs={0: torch.eye(d, dtype=DTYPE)},
        noise_covs={1: torch.zeros(d, d, dtype=DTYPE),
                    2: torch.eye(d, dtype=DTYPE),
                    3: torch.eye(d, dtype=DTYPE)},
    )


# ============================================================
# Test 1: A singular conditioning covariance Sigma_{Z,Z} raises the
# diagnostic ValueError (not an opaque PyTorch linear-algebra error).
# ============================================================


def test_singular_conditioning_raises_diagnostic_valueerror():
    K = _build_duplicated_node_K(d=2, seed=700)
    with pytest.raises(ValueError, match="Conditioning covariance"):
        conditional_mutual_information_from_k(K, A=[2], B=[3], C=[0, 1])


# ============================================================
# Test 2: jitter > 0 regularizes the conditioning block and rescues the
# singular case. Given V_0, nodes 2 and 3 see independent noises, so
# I(V_2; V_3 | V_0, V_1) == 0 up to the jitter perturbation.
# ============================================================


def test_jitter_regularizes_conditioning_block():
    K = _build_duplicated_node_K(d=2, seed=710)
    I = conditional_mutual_information_from_k(
        K, A=[2], B=[3], C=[0, 1], jitter=1e-9
    )
    assert torch.isfinite(I)
    assert abs(I.item()) < 1e-6, (
        f"expected I(V2; V3 | V0, V1) ~ 0, got {I.item():.3e}"
    )


# ============================================================
# Test 3: The Cholesky-based Schur solve agrees with an explicit
# generic-solve reference on a well-conditioned case.
# ============================================================


def test_cholesky_solve_matches_generic_solve_reference():
    d = 3
    K = _build_mac_K(d=d, seed=720)

    def assemble(rows, cols):
        return torch.cat(
            [torch.cat([get_K(K, r, c) for c in cols], dim=-1) for r in rows],
            dim=-2,
        )

    def cond_cov(A, Z):
        S_AA = assemble(A, A)
        if not Z:
            return S_AA
        S_AZ = assemble(A, Z)
        S_ZZ = assemble(Z, Z)
        return S_AA - S_AZ @ torch.linalg.solve(S_ZZ, S_AZ.mH)

    def logdet(M):
        M = 0.5 * (M + M.mH)
        L = torch.linalg.cholesky(M)
        return 2.0 * torch.log(torch.diagonal(L).real).sum()

    I_lib = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    I_ref = logdet(cond_cov([0], [1])) - logdet(cond_cov([0], [1, 2]))
    assert torch.allclose(I_lib, I_ref, atol=1e-10)


# ============================================================
# Test 4: Duplicate parent entries are rejected (they would silently
# double-count the edge contribution).
# ============================================================


def test_duplicate_parents_rejected():
    d = 2
    with pytest.raises(ValueError, match="duplicate"):
        compute_k_blocks_multiroot(
            num_nodes=2, roots=[0], parents={1: [0, 0]},
            edge_mats={(1, 0): _randn_complex(d, d, seed=730)},
            root_covs={0: torch.eye(d, dtype=DTYPE)},
            noise_covs={1: torch.eye(d, dtype=DTYPE)},
        )


# ============================================================
# Test 5: An edge declared in `parents` but missing from `edge_mats`
# raises ValueError (not KeyError).
# ============================================================


def test_missing_edge_mat_rejected():
    d = 2
    with pytest.raises(ValueError, match="edge_mats is missing"):
        compute_k_blocks_multiroot(
            num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
            edge_mats={(2, 0): _randn_complex(d, d, seed=740)},  # (2, 1) absent
            root_covs={0: torch.eye(d, dtype=DTYPE),
                       1: torch.eye(d, dtype=DTYPE)},
            noise_covs={2: torch.eye(d, dtype=DTYPE)},
        )
