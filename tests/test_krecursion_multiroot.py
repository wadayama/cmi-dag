"""Unit tests for cmi_dag.krecursion.compute_k_blocks_multiroot."""

from __future__ import annotations

import pytest
import torch

from cmi_dag.krecursion import compute_k_blocks_multiroot
from gdag_reference import compute_k_blocks  # single-root reference oracle

DTYPE = torch.complex128


def _randn_complex(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(*shape, dtype=torch.float64, generator=g)
    imag = torch.randn(*shape, dtype=torch.float64, generator=g)
    return torch.complex(real, imag)


def _hermitian_psd(d: int, *, seed: int) -> torch.Tensor:
    A = _randn_complex(d, d, seed=seed)
    return A @ A.mH + torch.eye(d, dtype=DTYPE)


# ============================================================
# Test 1: key coverage on a 2-user MAC (2 roots + 1 receiver).
# ============================================================


def test_mac_key_coverage():
    M = 3
    d = 2
    roots = [0, 1]
    parents = {2: [0, 1]}
    edge_mats = {
        (2, 0): _randn_complex(d, d, seed=1),
        (2, 1): _randn_complex(d, d, seed=2),
    }
    root_covs = {0: torch.eye(d, dtype=DTYPE), 1: torch.eye(d, dtype=DTYPE)}
    noise_covs = {2: torch.eye(d, dtype=DTYPE)}

    K = compute_k_blocks_multiroot(
        num_nodes=M, roots=roots, parents=parents,
        edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
    )

    expected_keys = {(j, k) for j in range(M) for k in range(j + 1)}
    assert set(K.keys()) == expected_keys


# ============================================================
# Test 2: root self-covariances and zero cross-blocks.
# ============================================================


def test_root_base_case():
    d = 3
    Sigma_0 = _hermitian_psd(d, seed=10)
    Sigma_1 = _hermitian_psd(d, seed=11)
    edge_mats = {(2, 0): _randn_complex(d, d, seed=20),
                 (2, 1): _randn_complex(d, d, seed=21)}
    root_covs = {0: Sigma_0, 1: Sigma_1}
    noise_covs = {2: torch.eye(d, dtype=DTYPE)}

    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
    )

    # K_{rr} == Sigma_r (symmetrized).
    assert torch.allclose(K[(0, 0)], 0.5 * (Sigma_0 + Sigma_0.mH))
    assert torch.allclose(K[(1, 1)], 0.5 * (Sigma_1 + Sigma_1.mH))
    # K_{1,0} == 0  (mutual independence of distinct roots).
    assert torch.allclose(K[(1, 0)], torch.zeros(d, d, dtype=DTYPE))


# ============================================================
# Test 3: single-root reduction agrees with parent compute_k_blocks.
# When K = 1, multi-root should match the parent on every K-block.
# ============================================================


def test_single_root_matches_parent():
    M = 3
    d = 2
    parents = {1: [0], 2: [1]}
    edge_mats = {
        (1, 0): _randn_complex(d, d, seed=30),
        (2, 1): _randn_complex(d, d, seed=31),
    }
    input_cov = _hermitian_psd(d, seed=32)
    noise_covs = {1: _hermitian_psd(d, seed=33), 2: _hermitian_psd(d, seed=34)}

    K_parent = compute_k_blocks(
        num_nodes=M, parents=parents, edge_mats=edge_mats,
        input_cov=input_cov, noise_covs=noise_covs,
    )
    K_child = compute_k_blocks_multiroot(
        num_nodes=M, roots=[0], parents=parents, edge_mats=edge_mats,
        root_covs={0: input_cov}, noise_covs=noise_covs,
    )

    assert set(K_parent.keys()) == set(K_child.keys())
    for key in K_parent:
        assert torch.allclose(K_parent[key], K_child[key], atol=1e-12), (
            f"mismatch at K{key}"
        )


# ============================================================
# Test 4: non-root self-covariance formula (MAC receiver).
# K_{YY} = A_{Y,0} Sigma_0 A_{Y,0}^H + A_{Y,1} Sigma_1 A_{Y,1}^H + Sigma_Y.
# ============================================================


def test_mac_self_block_formula():
    d = 2
    A0 = _randn_complex(d, d, seed=40)
    A1 = _randn_complex(d, d, seed=41)
    Sigma_0 = _hermitian_psd(d, seed=42)
    Sigma_1 = _hermitian_psd(d, seed=43)
    Sigma_Y = _hermitian_psd(d, seed=44)

    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): A0, (2, 1): A1},
        root_covs={0: Sigma_0, 1: Sigma_1},
        noise_covs={2: Sigma_Y},
    )

    expected = A0 @ Sigma_0 @ A0.mH + A1 @ Sigma_1 @ A1.mH + Sigma_Y
    expected = 0.5 * (expected + expected.mH)  # hermitianize
    assert torch.allclose(K[(2, 2)], expected, atol=1e-12)


# ============================================================
# Test 5: cross block K_{Y, X_0} = A_{Y,0} * Sigma_0.
# ============================================================


def test_mac_cross_block_to_root():
    d = 2
    A0 = _randn_complex(d, d, seed=50)
    A1 = _randn_complex(d, d, seed=51)
    Sigma_0 = _hermitian_psd(d, seed=52)
    Sigma_1 = _hermitian_psd(d, seed=53)

    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): A0, (2, 1): A1},
        root_covs={0: Sigma_0, 1: Sigma_1},
        noise_covs={2: torch.eye(d, dtype=DTYPE)},
    )

    # K_{2,0} = A0 * Sigma_0 + A1 * K_{1,0}  and  K_{1,0} = 0  ->  A0 * Sigma_0.
    assert torch.allclose(K[(2, 0)], A0 @ Sigma_0, atol=1e-12)
    assert torch.allclose(K[(2, 1)], A1 @ Sigma_1, atol=1e-12)


# ============================================================
# Test 6: differentiability through edge_mats.
# ============================================================


def test_differentiability_through_edge_mats():
    d = 2
    A0 = _randn_complex(d, d, seed=60).requires_grad_(True)
    A1 = _randn_complex(d, d, seed=61).requires_grad_(True)
    Sigma_0 = torch.eye(d, dtype=DTYPE)
    Sigma_1 = torch.eye(d, dtype=DTYPE)
    Sigma_Y = torch.eye(d, dtype=DTYPE)

    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): A0, (2, 1): A1},
        root_covs={0: Sigma_0, 1: Sigma_1},
        noise_covs={2: Sigma_Y},
    )
    loss = torch.real(torch.trace(K[(2, 2)]))
    loss.backward()
    assert A0.grad is not None
    assert A1.grad is not None
    assert torch.isfinite(A0.grad).all()
    assert torch.isfinite(A1.grad).all()


# ============================================================
# Test 7: invalid roots raise ValueError.
# ============================================================


def test_invalid_roots_raises():
    d = 2
    # roots = [0, 2]  is not the prefix {0, 1}.
    with pytest.raises(ValueError, match="prefix"):
        compute_k_blocks_multiroot(
            num_nodes=4, roots=[0, 2], parents={1: [0], 3: [2]},
            edge_mats={(1, 0): torch.eye(d, dtype=DTYPE),
                       (3, 2): torch.eye(d, dtype=DTYPE)},
            root_covs={0: torch.eye(d, dtype=DTYPE),
                       2: torch.eye(d, dtype=DTYPE)},
            noise_covs={1: torch.eye(d, dtype=DTYPE),
                        3: torch.eye(d, dtype=DTYPE)},
        )


def test_all_roots_no_non_root_raises():
    d = 2
    # num_roots == num_nodes: no non-root, no channel.
    with pytest.raises(ValueError, match="at least one non-root"):
        compute_k_blocks_multiroot(
            num_nodes=2, roots=[0, 1], parents={},
            edge_mats={},
            root_covs={0: torch.eye(d, dtype=DTYPE),
                       1: torch.eye(d, dtype=DTYPE)},
            noise_covs={},
        )
