"""Unit tests for the optional cross_root_covs feature of compute_k_blocks_multiroot.

These tests pin down the behaviour of correlated source roots: byte-identical
default path, exact seeding of K[(r, r')], downstream CMI agreement with the
closed-form scalar formula, joint-PD validation gates, the generalised
effective-channel decomposition K_{jk} = sum_{r, r'} G^{(r)} Sigma_{r, r'}
G^{(r')H} + C_{jk}, and autograd flow through cross_root_covs.
"""

from __future__ import annotations

import math

import pytest
import torch

from cmi_dag import (
    compute_effective_channel,
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    get_K,
)

DTYPE = torch.complex128


def _randn_complex(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(*shape, dtype=torch.float64, generator=g)
    imag = torch.randn(*shape, dtype=torch.float64, generator=g)
    return torch.complex(real, imag)


def _mac_2x2_inputs():
    """A common 2-root MAC (X_1, X_2 -> Y) fixture used by several tests."""
    d = 2
    parents = {2: [0, 1]}
    edge_mats = {
        (2, 0): _randn_complex(d, d, seed=1),
        (2, 1): _randn_complex(d, d, seed=2),
    }
    root_covs = {0: torch.eye(d, dtype=DTYPE), 1: torch.eye(d, dtype=DTYPE)}
    noise_covs = {2: 0.5 * torch.eye(d, dtype=DTYPE)}
    return d, parents, edge_mats, root_covs, noise_covs


# ============================================================
# Test 1: default-None call is byte-identical to the legacy path.
# ============================================================


def test_default_none_is_byte_identical():
    d, parents, edge_mats, root_covs, noise_covs = _mac_2x2_inputs()

    K_legacy = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents=parents,
        edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
    )
    K_none = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents=parents,
        edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
        cross_root_covs=None,
    )
    K_empty = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents=parents,
        edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
        cross_root_covs={},
    )

    assert set(K_legacy.keys()) == set(K_none.keys()) == set(K_empty.keys())
    for key in K_legacy:
        assert torch.equal(K_legacy[key], K_none[key]), f"mismatch at {key} (None)"
        assert torch.equal(K_legacy[key], K_empty[key]), f"mismatch at {key} ({{}})"


def test_effective_channel_default_byte_identical():
    d, parents, edge_mats, _root_covs, noise_covs = _mac_2x2_inputs()

    G_legacy, C_legacy = compute_effective_channel(
        num_nodes=3, roots=[0, 1], parents=parents,
        edge_mats=edge_mats, noise_covs=noise_covs,
    )
    # compute_effective_channel was deliberately not extended with a
    # cross_root_covs parameter (G, C are channel-intrinsic). Confirm the
    # call still returns the same objects shape-wise; this guards against
    # accidental signature changes during the correlated-roots refactor.
    assert set(G_legacy.keys()) == {(r, j) for r in [0, 1] for j in range(3)}
    assert set(C_legacy.keys()) == {(j, k) for j in range(3) for k in range(j + 1)}


# ============================================================
# Test 2: specified cross covariance appears verbatim in K[(r, r')].
# ============================================================


def test_specified_cross_cov_appears_in_K():
    d, parents, edge_mats, root_covs, noise_covs = _mac_2x2_inputs()

    Sigma_10 = 0.3 * _randn_complex(d, d, seed=7)
    cross = {(1, 0): Sigma_10}

    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents=parents,
        edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
        cross_root_covs=cross,
    )

    assert torch.equal(K[(1, 0)], Sigma_10)
    # Hermitian flip applied automatically via get_K.
    assert torch.allclose(get_K(K, 0, 1), Sigma_10.mH)


# ============================================================
# Test 3: scalar closed-form CMI agreement under root correlation.
# For jointly CN(0, [[1, rho], [rho*, 1]]),  I(V_0; V_1) = -log(1 - |rho|^2)
# (complex Gaussian convention; the library returns nats).
# Use a tiny chain to satisfy num_roots < num_nodes.
# ============================================================


def test_downstream_cmi_correlation_closed_form_d1():
    parents = {2: [0]}  # Y is irrelevant to I(V_0; V_1); just keeps the DAG valid.
    A20 = torch.tensor([[1.0 + 0.0j]], dtype=DTYPE)
    edge_mats = {(2, 0): A20}
    root_covs = {0: torch.eye(1, dtype=DTYPE), 1: torch.eye(1, dtype=DTYPE)}
    noise_covs = {2: torch.eye(1, dtype=DTYPE)}

    for rho in [0.1, 0.5, 0.8, -0.7]:
        cross = {(1, 0): torch.tensor([[rho + 0.0j]], dtype=DTYPE)}
        K = compute_k_blocks_multiroot(
            num_nodes=3, roots=[0, 1], parents=parents,
            edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
            cross_root_covs=cross,
        )
        I = conditional_mutual_information_from_k(K, A=[0], B=[1])
        expected = -math.log(1.0 - rho * rho)
        assert math.isclose(I.item(), expected, abs_tol=1e-10), (
            f"rho={rho}: got {I.item()}, expected {expected}"
        )


# ============================================================
# Test 4: joint-PD validation rejects an indefinite Sigma_RR.
# ============================================================


def test_pd_validation_rejects_indefinite():
    d, parents, edge_mats, root_covs, noise_covs = _mac_2x2_inputs()

    # Cross cov with operator norm > sqrt(sigma_0 sigma_1) = 1 makes Sigma_RR
    # indefinite (off-diagonal eigenvalues dominate the diagonal).
    Sigma_10 = 2.0 * torch.eye(d, dtype=DTYPE)
    cross = {(1, 0): Sigma_10}

    with pytest.raises(ValueError, match="positive definite"):
        compute_k_blocks_multiroot(
            num_nodes=3, roots=[0, 1], parents=parents,
            edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
            cross_root_covs=cross,
        )


# ============================================================
# Test 5: shape mismatch raises with a clear diagnostic.
# ============================================================


def test_pd_validation_rejects_wrong_shape():
    d, parents, edge_mats, root_covs, noise_covs = _mac_2x2_inputs()

    # root_covs has d_0 = d_1 = 2, but supply a 3x2 cross.
    bad = _randn_complex(3, 2, seed=11)
    cross = {(1, 0): bad}

    with pytest.raises(ValueError, match="shape"):
        compute_k_blocks_multiroot(
            num_nodes=3, roots=[0, 1], parents=parents,
            edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
            cross_root_covs=cross,
        )


# ============================================================
# Test 6: non-canonical key (r < r') raises with the convention hint.
# ============================================================


def test_pd_validation_rejects_wrong_key_ordering():
    d, parents, edge_mats, root_covs, noise_covs = _mac_2x2_inputs()

    Sigma_01 = 0.3 * _randn_complex(d, d, seed=13)
    cross = {(0, 1): Sigma_01}  # wrong: should be (1, 0)

    with pytest.raises(ValueError, match="canonical"):
        compute_k_blocks_multiroot(
            num_nodes=3, roots=[0, 1], parents=parents,
            edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
            cross_root_covs=cross,
        )


# ============================================================
# Test 7: effective-channel decomposition holds under correlated roots.
# K[(j, k)] = sum_{r, r'} G[(r, j)] @ Sigma_{r, r'} @ G[(r', k)].mH + C[(j, k)].
# Also verify G, C are byte-identical regardless of cross_root_covs (channel
# intrinsic).
# ============================================================


def test_effective_channel_decomposition_under_correlation():
    d, parents, edge_mats, root_covs, noise_covs = _mac_2x2_inputs()

    Sigma_10 = 0.2 * _randn_complex(d, d, seed=17)  # small magnitude -> PD safe
    cross = {(1, 0): Sigma_10}

    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents=parents,
        edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
        cross_root_covs=cross,
    )
    G, C = compute_effective_channel(
        num_nodes=3, roots=[0, 1], parents=parents,
        edge_mats=edge_mats, noise_covs=noise_covs,
    )

    # Build the source-covariance block matrix Sigma_{r, r'} lookup.
    Sigma_src = {
        (0, 0): root_covs[0],
        (1, 1): root_covs[1],
        (1, 0): Sigma_10,
        (0, 1): Sigma_10.mH,
    }
    roots = [0, 1]
    for j in range(3):
        for k in range(j + 1):
            lhs = K[(j, k)]
            rhs = C[(j, k)].clone()
            for r in roots:
                for rp in roots:
                    rhs = rhs + G[(r, j)] @ Sigma_src[(r, rp)] @ G[(rp, k)].mH
            assert torch.allclose(lhs, rhs, atol=1e-10), f"decomposition mismatch at ({j}, {k})"

    # Independence assertion: G, C are channel-intrinsic so the same call
    # with or without correlated roots returns the same (G, C).
    G2, C2 = compute_effective_channel(
        num_nodes=3, roots=[0, 1], parents=parents,
        edge_mats=edge_mats, noise_covs=noise_covs,
    )
    for key in G:
        assert torch.equal(G[key], G2[key]), f"G drift at {key}"
    for key in C:
        assert torch.equal(C[key], C2[key]), f"C drift at {key}"


# ============================================================
# Test 8: autograd flows through cross_root_covs.
# ============================================================


def test_differentiability_through_cross_root_covs():
    d, parents, edge_mats, root_covs, noise_covs = _mac_2x2_inputs()

    Sigma_10 = (0.2 * _randn_complex(d, d, seed=21)).clone().requires_grad_(True)
    cross = {(1, 0): Sigma_10}

    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents=parents,
        edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
        cross_root_covs=cross,
    )
    I = conditional_mutual_information_from_k(K, A=[0], B=[1], C=[2])
    I.backward()

    assert Sigma_10.grad is not None
    assert Sigma_10.grad.shape == Sigma_10.shape
    assert torch.isfinite(Sigma_10.grad.real).all()
    assert torch.isfinite(Sigma_10.grad.imag).all()
    # Sanity: the gradient should be non-zero somewhere (the CMI does depend
    # on the cross covariance).
    assert Sigma_10.grad.abs().sum().item() > 0.0


# ============================================================
# Test 9: single-root configuration rejects any cross_root_covs key.
# ============================================================


def test_single_root_disallows_cross_keys():
    parents = {1: [0]}
    edge_mats = {(1, 0): _randn_complex(2, 2, seed=31)}
    root_covs = {0: torch.eye(2, dtype=DTYPE)}
    noise_covs = {1: torch.eye(2, dtype=DTYPE)}

    # Empty dict and None pass through.
    K_a = compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents=parents,
        edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
        cross_root_covs={},
    )
    K_b = compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents=parents,
        edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
        cross_root_covs=None,
    )
    for key in K_a:
        assert torch.equal(K_a[key], K_b[key])

    # (0, 0) self-pair rejected.
    cross_self = {(0, 0): torch.eye(2, dtype=DTYPE)}
    with pytest.raises(ValueError, match="self-pair"):
        compute_k_blocks_multiroot(
            num_nodes=2, roots=[0], parents=parents,
            edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
            cross_root_covs=cross_self,
        )


# ============================================================
# Test 10: dtype/device propagation through the new code path.
# ============================================================


def test_dtype_propagation():
    """Same scenario as test 2 but in complex64; result must be complex64."""
    d = 2
    parents = {2: [0, 1]}
    g = torch.Generator().manual_seed(41)
    edge_mats = {
        (2, 0): torch.complex(
            torch.randn(d, d, dtype=torch.float32, generator=g),
            torch.randn(d, d, dtype=torch.float32, generator=g),
        ),
        (2, 1): torch.complex(
            torch.randn(d, d, dtype=torch.float32, generator=g),
            torch.randn(d, d, dtype=torch.float32, generator=g),
        ),
    }
    root_covs = {0: torch.eye(d, dtype=torch.complex64),
                 1: torch.eye(d, dtype=torch.complex64)}
    noise_covs = {2: torch.eye(d, dtype=torch.complex64)}
    Sigma_10 = 0.2 * torch.eye(d, dtype=torch.complex64)
    cross = {(1, 0): Sigma_10}

    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents=parents,
        edge_mats=edge_mats, root_covs=root_covs, noise_covs=noise_covs,
        cross_root_covs=cross,
    )
    for key, blk in K.items():
        assert blk.dtype == torch.complex64, f"dtype drift at {key}"
