"""Unit tests for cmi_dag.krecursion.compute_effective_channel (multi-root)."""

from __future__ import annotations

import pytest
import torch

from cmi_dag.information import conditional_mutual_information_from_k, logdet_hpd
from cmi_dag.krecursion import compute_effective_channel, compute_k_blocks_multiroot
from gdag_reference import (  # single-root reference oracle
    compute_effective_channel as compute_effective_channel_single_root,
)

DTYPE = torch.complex128


def _randn_complex(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(*shape, dtype=torch.float64, generator=g)
    imag = torch.randn(*shape, dtype=torch.float64, generator=g)
    return torch.complex(real, imag)


def _hermitian_psd(d: int, *, seed: int) -> torch.Tensor:
    A = _randn_complex(d, d, seed=seed)
    return A @ A.mH + torch.eye(d, dtype=DTYPE)


def _mac_problem(d: int = 2):
    """2-user MAC: roots {0, 1} -> receiver 2."""
    M, roots = 3, [0, 1]
    parents = {2: [0, 1]}
    edge_mats = {
        (2, 0): _randn_complex(d, d, seed=1),
        (2, 1): _randn_complex(d, d, seed=2),
    }
    noise_covs = {2: _hermitian_psd(d, seed=12)}
    return M, roots, parents, edge_mats, noise_covs


def _relay_problem(d: int = 2):
    """Roots {0, 1}; 0 -> relay 2 -> sink 3, 1 -> sink 3 directly."""
    M, roots = 4, [0, 1]
    parents = {2: [0], 3: [1, 2]}
    edge_mats = {
        (2, 0): _randn_complex(d, d, seed=1),
        (3, 1): _randn_complex(d, d, seed=2),
        (3, 2): _randn_complex(d, d, seed=3),
    }
    noise_covs = {2: _hermitian_psd(d, seed=12), 3: _hermitian_psd(d, seed=13)}
    return M, roots, parents, edge_mats, noise_covs


def test_effective_channel_base_case():
    M, roots, parents, edge_mats, noise_covs = _mac_problem(d=2)
    G, C = compute_effective_channel(M, roots, parents, edge_mats, noise_covs)
    for r in roots:
        assert torch.allclose(G[(r, r)], torch.eye(2, dtype=DTYPE), atol=1e-12)
        for r2 in roots:
            if r2 != r:
                assert torch.allclose(
                    G[(r, r2)], torch.zeros(2, 2, dtype=DTYPE), atol=1e-12
                )
        assert torch.allclose(C[(r, r)], torch.zeros(2, 2, dtype=DTYPE), atol=1e-12)


def _check_identity(M, roots, parents, edge_mats, noise_covs, seed):
    root_covs = {r: _hermitian_psd(2, seed=seed + r) for r in roots}
    K = compute_k_blocks_multiroot(
        M, roots, parents, edge_mats, root_covs, noise_covs
    )
    G, C = compute_effective_channel(M, roots, parents, edge_mats, noise_covs)
    for (j, k) in K:
        rebuilt = C[(j, k)].clone()
        for r in roots:
            rebuilt = rebuilt + G[(r, j)] @ root_covs[r] @ G[(r, k)].mH
        assert torch.allclose(K[(j, k)], rebuilt, atol=1e-10), (
            f"K_{{{j},{k}}} != sum_r G_j^(r) Sigma_r G_k^(r)^H + C_{{{j},{k}}}."
        )


def test_K_equals_sum_G_sigma_G_plus_C_mac():
    _check_identity(*_mac_problem(d=2), seed=20)


def test_K_equals_sum_G_sigma_G_plus_C_relay():
    _check_identity(*_relay_problem(d=2), seed=30)


def test_single_root_reduction_matches_oracle():
    # Single-root chain: cmi multi-root with roots=[0] must reproduce the
    # gaussian-dag single-root effective channel block-for-block.
    M, d = 3, 2
    parents = {1: [0], 2: [1]}
    edge_mats = {
        (1, 0): _randn_complex(d, d, seed=1),
        (2, 1): _randn_complex(d, d, seed=2),
    }
    noise_covs = {1: _hermitian_psd(d, seed=11), 2: _hermitian_psd(d, seed=12)}

    G_multi, C_multi = compute_effective_channel(
        M, [0], parents, edge_mats, noise_covs
    )
    G_ref, C_ref = compute_effective_channel_single_root(
        M, parents, edge_mats, noise_covs
    )
    for j in range(M):
        assert torch.allclose(G_multi[(0, j)], G_ref[j], atol=1e-12)
    for key in C_ref:
        assert torch.allclose(C_multi[key], C_ref[key], atol=1e-12)


def test_effective_channel_mi_identity_mac():
    # Full-input MI I(X_0, X_1; Y) via the effective channel must match
    # conditional_mutual_information_from_k(A=roots, B=[sink], C=[]).
    M, roots, parents, edge_mats, noise_covs = _mac_problem(d=2)
    root_covs = {r: _hermitian_psd(2, seed=40 + r) for r in roots}
    K = compute_k_blocks_multiroot(
        M, roots, parents, edge_mats, root_covs, noise_covs
    )
    I_ref = conditional_mutual_information_from_k(K, A=roots, B=[M - 1], C=[])

    G, C = compute_effective_channel(M, roots, parents, edge_mats, noise_covs)
    sink = M - 1
    cov_signal = C[(sink, sink)].clone()
    for r in roots:
        cov_signal = cov_signal + G[(r, sink)] @ root_covs[r] @ G[(r, sink)].mH
    I_eff = logdet_hpd(cov_signal) - logdet_hpd(C[(sink, sink)])
    assert torch.allclose(I_eff, I_ref, atol=1e-10)


def test_effective_channel_dim_inference_and_override():
    # Non-square sources: d_0 = 2, d_1 = 3, receiver dim 2.
    M, roots = 3, [0, 1]
    parents = {2: [0, 1]}
    edge_mats = {
        (2, 0): _randn_complex(2, 2, seed=1),  # (d_2, d_0) = (2, 2)
        (2, 1): _randn_complex(2, 3, seed=2),  # (d_2, d_1) = (2, 3)
    }
    noise_covs = {2: _hermitian_psd(2, seed=12)}

    G_inf, _ = compute_effective_channel(M, roots, parents, edge_mats, noise_covs)
    assert torch.allclose(G_inf[(0, 0)], torch.eye(2, dtype=DTYPE), atol=1e-12)
    assert torch.allclose(G_inf[(1, 1)], torch.eye(3, dtype=DTYPE), atol=1e-12)
    assert G_inf[(0, 2)].shape == (2, 2)
    assert G_inf[(1, 2)].shape == (2, 3)

    G_exp, _ = compute_effective_channel(
        M, roots, parents, edge_mats, noise_covs, source_dims={0: 2, 1: 3}
    )
    for key in G_inf:
        assert torch.allclose(G_inf[key], G_exp[key], atol=1e-12)


def test_effective_channel_device_agnostic_smoke():
    M, roots, parents, edge_mats, noise_covs = _relay_problem(d=2)
    G, C = compute_effective_channel(M, roots, parents, edge_mats, noise_covs)
    ref = edge_mats[(2, 0)]
    for t in list(G.values()) + list(C.values()):
        assert t.dtype == ref.dtype
        assert t.device == ref.device


def test_effective_channel_errors():
    # roots not the prefix {0, ..., K-1} -> ValueError.
    with pytest.raises(ValueError):
        compute_effective_channel(
            3, [0, 2], {2: [0]}, {(2, 0): _randn_complex(2, 2, seed=1)},
            {2: _hermitian_psd(2, seed=9)},
        )
    # A root with no outgoing edge and no source_dims entry -> ValueError.
    with pytest.raises(ValueError):
        # root 1 has no outgoing edge; only root 0 feeds node 2.
        compute_effective_channel(
            3, [0, 1], {2: [0]}, {(2, 0): _randn_complex(2, 2, seed=1)},
            {2: _hermitian_psd(2, seed=9)},
        )
