"""Unit tests for cmi_dag.information.conditional_mutual_information_from_k."""

from __future__ import annotations

import pytest
import torch

from cmi_dag.information import conditional_mutual_information_from_k
from cmi_dag.krecursion import compute_k_blocks_multiroot
from gdag_reference import (  # single-root reference oracles
    compute_k_blocks,
    mutual_information_from_k,
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


# ============================================================
# Test 1: Empty C reduces to unconditional MI, agrees with parent.
# ============================================================


def test_unconditional_matches_parent_single_link():
    d = 2
    parents = {1: [0]}
    edge_mats = {(1, 0): _randn_complex(d, d, seed=100)}
    input_cov = _hermitian_psd(d, seed=101)
    noise_covs = {1: _hermitian_psd(d, seed=102)}

    K_parent = compute_k_blocks(
        num_nodes=2, parents=parents, edge_mats=edge_mats,
        input_cov=input_cov, noise_covs=noise_covs,
    )
    K_child = compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents=parents, edge_mats=edge_mats,
        root_covs={0: input_cov}, noise_covs=noise_covs,
    )

    I_parent = mutual_information_from_k(K_parent, output_node=1, input_node=0)
    I_child = conditional_mutual_information_from_k(K_child, A=[0], B=[1], C=[])

    assert torch.allclose(I_parent, I_child, atol=1e-12)


# ============================================================
# Test 2: Non-negativity: I(A;B|C) >= 0.
# ============================================================


def test_cmi_nonnegative():
    K = _build_mac_K(d=2, seed=200)
    I1 = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    I2 = conditional_mutual_information_from_k(K, A=[1], B=[2], C=[0])
    I_sigma = conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[])
    assert I1.item() >= -1e-12
    assert I2.item() >= -1e-12
    assert I_sigma.item() >= -1e-12


# ============================================================
# Test 3: Symmetry I(A; B | C) == I(B; A | C).
# ============================================================


def test_cmi_symmetry():
    K = _build_mac_K(d=2, seed=300)
    I_ab = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    I_ba = conditional_mutual_information_from_k(K, A=[2], B=[0], C=[1])
    assert torch.allclose(I_ab, I_ba, atol=1e-10)


# ============================================================
# Test 4: Pentagon chain rule
# I(X1, X2; Y) = I(X1; Y | X2) + I(X2; Y).
# Both sides computed by the same function.
# ============================================================


def test_pentagon_chain_rule():
    K = _build_mac_K(d=3, seed=400)
    I_sigma = conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[])
    I_1_given_2 = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    I_2 = conditional_mutual_information_from_k(K, A=[1], B=[2], C=[])
    assert torch.allclose(I_sigma, I_1_given_2 + I_2, atol=1e-10)


# ============================================================
# General chain rule: I(A; B, C) = I(A; B) + I(A; C | B), for disjoint A, B, C.
# Apply it on the MAC: A = {X1}, B = {Y}, C = {X2}.
# Note I(X1; X2) = 0 (independent roots), but I(X1; X2 | Y) > 0 because
# observing Y induces correlation between the two transmitters.
# ============================================================


def test_general_chain_rule_mac():
    K = _build_mac_K(d=3, seed=450)
    # LHS: I(X1; Y, X2)
    lhs = conditional_mutual_information_from_k(K, A=[0], B=[1, 2], C=[])
    # RHS: I(X1; Y) + I(X1; X2 | Y)
    I_X1_Y = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[])
    I_X1_X2_given_Y = conditional_mutual_information_from_k(
        K, A=[0], B=[1], C=[2]
    )
    rhs = I_X1_Y + I_X1_X2_given_Y
    assert torch.allclose(lhs, rhs, atol=1e-10), (
        f"chain rule failed: LHS = {lhs.item():.6e}, RHS = {rhs.item():.6e}"
    )


def test_general_chain_rule_swap():
    """Same chain rule with the roles of B and C swapped:
    I(A; B, C) = I(A; C) + I(A; B | C). Take A = {X1}, B = {X2}, C = {Y}."""
    K = _build_mac_K(d=2, seed=460)
    lhs = conditional_mutual_information_from_k(K, A=[0], B=[1, 2], C=[])
    I_X1_Y = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[])
    I_X1_X2_given_Y = conditional_mutual_information_from_k(
        K, A=[0], B=[1], C=[2]
    )
    rhs = I_X1_Y + I_X1_X2_given_Y
    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_independent_roots_have_zero_unconditional_mi():
    """Sanity check feeding into the chain rule: I(X1; X2) = 0 for two
    independent roots (the K-recursion base case is K_{X1, X2} = 0)."""
    K = _build_mac_K(d=3, seed=470)
    I_X1_X2 = conditional_mutual_information_from_k(K, A=[0], B=[1], C=[])
    assert abs(I_X1_X2.item()) < 1e-10, (
        f"expected I(X1; X2) == 0 for independent roots, got {I_X1_X2.item():.3e}"
    )


# ============================================================
# Test 5: Differentiability through edge_mats (precoder gradient).
# ============================================================


def test_cmi_gradient_through_precoder():
    d = 2
    F = _randn_complex(d, d, seed=500).requires_grad_(True)
    H = _randn_complex(d, d, seed=501)
    A0 = H @ F
    A1 = torch.eye(d, dtype=DTYPE)

    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): A0, (2, 1): A1},
        root_covs={0: torch.eye(d, dtype=DTYPE),
                   1: torch.eye(d, dtype=DTYPE)},
        noise_covs={2: torch.eye(d, dtype=DTYPE)},
    )
    I = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    I.backward()
    assert F.grad is not None
    assert torch.isfinite(F.grad).all()


# ============================================================
# Test 6: Disjointness validation.
# ============================================================


def test_disjointness_required():
    K = _build_mac_K(d=2, seed=600)
    with pytest.raises(ValueError, match="disjoint"):
        conditional_mutual_information_from_k(K, A=[0], B=[0, 1], C=[])


def test_nonempty_required():
    K = _build_mac_K(d=2, seed=601)
    with pytest.raises(ValueError, match="non-empty"):
        conditional_mutual_information_from_k(K, A=[], B=[2], C=[])


# ============================================================
# Test 7: batch-safety -- a leading config batch dimension yields one CMI per
# element, matching per-element evaluation (exercises batched logdet_hpd and the
# conditional-covariance Schur solve over the batch).
# ============================================================


def test_logdet_hpd_batched():
    from cmi_dag.information import logdet_hpd

    d, B = 3, 5
    mats = torch.stack([_hermitian_psd(d, seed=700 + b) for b in range(B)])
    out = logdet_hpd(mats)
    assert out.shape == (B,)
    expected = torch.linalg.slogdet(mats).logabsdet
    assert torch.allclose(out, expected, atol=1e-10)
    for b in range(B):
        assert torch.allclose(out[b], logdet_hpd(mats[b]), atol=1e-12)


def test_cmi_batched_matches_per_element():
    d, B = 2, 4
    A0 = torch.stack([_randn_complex(d, d, seed=300 + b) for b in range(B)])
    A1 = torch.stack([_randn_complex(d, d, seed=400 + b) for b in range(B)])
    r0 = torch.eye(d, dtype=DTYPE)
    r1 = torch.eye(d, dtype=DTYPE)
    nz = torch.eye(d, dtype=DTYPE)

    def build(a0, a1):
        return compute_k_blocks_multiroot(
            num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
            edge_mats={(2, 0): a0, (2, 1): a1},
            root_covs={0: r0, 1: r1}, noise_covs={2: nz},
        )

    # Conditioning set C=[1] exercises the conditional-covariance Schur path.
    cmi_b = conditional_mutual_information_from_k(build(A0, A1), A=[0], B=[2], C=[1])
    assert cmi_b.shape == (B,)
    for b in range(B):
        cmi_s = conditional_mutual_information_from_k(
            build(A0[b], A1[b]), A=[0], B=[2], C=[1])
        assert torch.allclose(cmi_b[b], cmi_s, atol=1e-10)
