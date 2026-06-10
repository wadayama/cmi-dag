"""Unit tests for cmi_dag.information.conditional_differential_entropy_from_k.

The conditional differential entropy of a circular complex Gaussian is

    h(V_A | V_C) = log det Sigma_{A|C} + d_A * log(pi e)   (nats),

the "one-Schur, one-log-det" half of the conditional MI pipeline
    I(V_A; V_B | V_C) = h(V_A | V_C) - h(V_A | V_BC).

Reference log-dets here are computed independently via torch.linalg.slogdet
(not the library's own logdet_hpd) so the closed-form checks are genuine
cross-validations rather than tautologies.
"""

from __future__ import annotations

import math

import pytest
import torch

from cmi_dag import (
    compute_k_blocks_multiroot,
    conditional_differential_entropy_from_k,
    conditional_mutual_information_from_k,
)

DTYPE = torch.complex128

# Per-(complex-)dimension differential-entropy constant, in nats.
LOG_PI_E = math.log(math.pi * math.e)


def _randn_complex(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(*shape, dtype=torch.float64, generator=g)
    imag = torch.randn(*shape, dtype=torch.float64, generator=g)
    return torch.complex(real, imag)


def _hermitian_pd(d: int, *, seed: int) -> torch.Tensor:
    """Random Hermitian positive-definite d x d matrix."""
    A = _randn_complex(d, d, seed=seed)
    return A @ A.mH + torch.eye(d, dtype=DTYPE)


def _ref_logdet(A: torch.Tensor) -> torch.Tensor:
    """Independent reference log det for Hermitian PD A (real, nats).

    Uses slogdet rather than Cholesky so it is not the same code path as the
    library's logdet_hpd. For Hermitian PD A the determinant is real and
    positive, so logabsdet == log det.
    """
    sign, logabsdet = torch.linalg.slogdet(A)
    return logabsdet


def _build_mac_K(d: int, *, seed: int):
    """2-user MIMO MAC (roots 0,1 -> node 2) with random PD root/noise covs."""
    A0 = _randn_complex(d, d, seed=seed)
    A1 = _randn_complex(d, d, seed=seed + 1)
    return compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): A0, (2, 1): A1},
        root_covs={0: _hermitian_pd(d, seed=seed + 2),
                   1: _hermitian_pd(d, seed=seed + 3)},
        noise_covs={2: _hermitian_pd(d, seed=seed + 4)},
    )


# ============================================================
# Test 1: Marginal entropy closed form h(V_A) = log det Sigma + d * log(pi e).
# ============================================================


def test_marginal_entropy_closed_form():
    """For a single root node with covariance Sigma, the marginal entropy is
    log det Sigma + d * log(pi e), checked against an independent slogdet."""
    d = 3
    Sigma = _hermitian_pd(d, seed=700)
    K = compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents={1: [0]},
        edge_mats={(1, 0): _randn_complex(d, d, seed=701)},
        root_covs={0: Sigma},
        noise_covs={1: _hermitian_pd(d, seed=702)},
    )
    h_lib = conditional_differential_entropy_from_k(K, A=[0])
    h_ref = _ref_logdet(Sigma) + d * LOG_PI_E
    assert torch.allclose(h_lib, h_ref, atol=1e-10)


def test_marginal_entropy_of_output_node():
    """Same closed form, but for the (non-root) output node Y: its marginal
    covariance is read straight from K[(1, 1)]."""
    d = 2
    K = compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents={1: [0]},
        edge_mats={(1, 0): _randn_complex(d, d, seed=710)},
        root_covs={0: _hermitian_pd(d, seed=711)},
        noise_covs={1: _hermitian_pd(d, seed=712)},
    )
    h_lib = conditional_differential_entropy_from_k(K, A=[1])
    h_ref = _ref_logdet(K[(1, 1)]) + d * LOG_PI_E
    assert torch.allclose(h_lib, h_ref, atol=1e-10)


# ============================================================
# Test 2: Empty C (explicit and default) reduces to marginal entropy.
# ============================================================


def test_empty_C_reduces_to_marginal():
    K = _build_mac_K(d=2, seed=720)
    h_default = conditional_differential_entropy_from_k(K, A=[0])
    h_empty_tuple = conditional_differential_entropy_from_k(K, A=[0], C=())
    h_empty_list = conditional_differential_entropy_from_k(K, A=[0], C=[])
    assert torch.allclose(h_default, h_empty_tuple, atol=0.0)
    assert torch.allclose(h_default, h_empty_list, atol=0.0)


# ============================================================
# Test 3: Entropy chain rule h(A, C) = h(A | C) + h(C).
# Take A = {X1}, C = {Y}: conditioning on the (correlated) output is nontrivial.
# ============================================================


def test_entropy_chain_rule():
    K = _build_mac_K(d=2, seed=730)
    h_joint = conditional_differential_entropy_from_k(K, A=[0, 2])
    h_A_given_C = conditional_differential_entropy_from_k(K, A=[0], C=[2])
    h_C = conditional_differential_entropy_from_k(K, A=[2])
    assert torch.allclose(h_joint, h_A_given_C + h_C, atol=1e-10)


def test_entropy_chain_rule_three_way():
    """h(A, B, C) = h(A | B, C) + h(B | C) + h(C) on the full MAC node set."""
    K = _build_mac_K(d=3, seed=740)
    h_all = conditional_differential_entropy_from_k(K, A=[0, 1, 2])
    h_0_given_12 = conditional_differential_entropy_from_k(K, A=[0], C=[1, 2])
    h_1_given_2 = conditional_differential_entropy_from_k(K, A=[1], C=[2])
    h_2 = conditional_differential_entropy_from_k(K, A=[2])
    assert torch.allclose(h_all, h_0_given_12 + h_1_given_2 + h_2, atol=1e-10)


# ============================================================
# Test 4: Consistency with CMI  I(A;B|C) = h(A|C) - h(A|BC).
# ============================================================


def test_cmi_equals_entropy_difference():
    K = _build_mac_K(d=2, seed=750)
    for A, B, C in [([0], [2], [1]), ([1], [2], [0]), ([0], [1, 2], [])]:
        I_cmi = conditional_mutual_information_from_k(K, A=A, B=B, C=C)
        h_A_C = conditional_differential_entropy_from_k(K, A=A, C=C)
        h_A_BC = conditional_differential_entropy_from_k(
            K, A=A, C=sorted(B + C)
        )
        assert torch.allclose(I_cmi, h_A_C - h_A_BC, atol=1e-10), (
            f"CMI mismatch for A={A}, B={B}, C={C}"
        )


# ============================================================
# Test 5: Differentiability through a precoder; constant has zero gradient.
# ============================================================


def test_entropy_gradient_through_precoder():
    d = 2
    F = _randn_complex(d, d, seed=760).requires_grad_(True)
    H = _randn_complex(d, d, seed=761)
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): H @ F, (2, 1): torch.eye(d, dtype=DTYPE)},
        root_covs={0: torch.eye(d, dtype=DTYPE),
                   1: torch.eye(d, dtype=DTYPE)},
        noise_covs={2: torch.eye(d, dtype=DTYPE)},
    )
    h = conditional_differential_entropy_from_k(K, A=[2], C=[1])
    h.backward()
    assert F.grad is not None
    assert torch.isfinite(F.grad).all()


def test_constant_term_does_not_affect_gradient():
    """The additive d_A * log(pi e) is a pure constant, so the gradient of the
    entropy equals the gradient of the bare log-det term."""
    d = 2
    H = _randn_complex(d, d, seed=770)

    F1 = _randn_complex(d, d, seed=771).requires_grad_(True)
    K1 = compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents={1: [0]},
        edge_mats={(1, 0): H @ F1},
        root_covs={0: torch.eye(d, dtype=DTYPE)},
        noise_covs={1: torch.eye(d, dtype=DTYPE)},
    )
    h = conditional_differential_entropy_from_k(K1, A=[1])
    h.backward()

    # Same graph, but objective is just log det Sigma_Y (no constant).
    F2 = _randn_complex(d, d, seed=771).requires_grad_(True)
    K2 = compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents={1: [0]},
        edge_mats={(1, 0): H @ F2},
        root_covs={0: torch.eye(d, dtype=DTYPE)},
        noise_covs={1: torch.eye(d, dtype=DTYPE)},
    )
    logdet_only = _ref_logdet(K2[(1, 1)])
    logdet_only.backward()

    assert torch.allclose(F1.grad, F2.grad, atol=1e-9)


# ============================================================
# Test 6: The additive constant is exactly d_A * log(pi e), with d_A the sum
# of per-node dimensions (verified with heterogeneous node dimensions).
# ============================================================


def test_dA_constant_with_heterogeneous_dims():
    """Roots of different dimension (d_0=2, d_1=3) feeding a d_2=4 sink.
    h(A) - log det Sigma_{A,A} must equal (d_0 + d_1) * log(pi e)."""
    d0, d1, d2 = 2, 3, 4
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): _randn_complex(d2, d0, seed=780),
                   (2, 1): _randn_complex(d2, d1, seed=781)},
        root_covs={0: _hermitian_pd(d0, seed=782),
                   1: _hermitian_pd(d1, seed=783)},
        noise_covs={2: _hermitian_pd(d2, seed=784)},
    )
    h = conditional_differential_entropy_from_k(K, A=[0, 1])

    # Assemble Sigma_{[0,1],[0,1]} for the independent-reference log-det.
    top = torch.cat([K[(0, 0)], K[(1, 0)].mH], dim=-1)
    bot = torch.cat([K[(1, 0)], K[(1, 1)]], dim=-1)
    Sigma_AA = torch.cat([top, bot], dim=-2)

    constant = (h - _ref_logdet(Sigma_AA)).item()
    assert math.isclose(constant, (d0 + d1) * LOG_PI_E, rel_tol=0, abs_tol=1e-9)


# ============================================================
# Test 7: jitter is forwarded to the log-det.
# ============================================================


def test_jitter_propagates_to_logdet():
    d = 3
    Sigma = _hermitian_pd(d, seed=790)
    K = compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents={1: [0]},
        edge_mats={(1, 0): _randn_complex(d, d, seed=791)},
        root_covs={0: Sigma},
        noise_covs={1: _hermitian_pd(d, seed=792)},
    )
    jitter = 1e-3
    h_lib = conditional_differential_entropy_from_k(K, A=[0], jitter=jitter)
    eye = torch.eye(d, dtype=DTYPE)
    h_ref = _ref_logdet(Sigma + jitter * eye) + d * LOG_PI_E
    assert torch.allclose(h_lib, h_ref, atol=1e-10)
    # And jitter actually changed the value relative to jitter=0.
    h_nojit = conditional_differential_entropy_from_k(K, A=[0])
    assert not torch.allclose(h_lib, h_nojit, atol=1e-8)


# ============================================================
# Test 8: Return type is a real scalar on the K-blocks' device.
# ============================================================


def test_return_is_real_scalar():
    K = _build_mac_K(d=2, seed=800)
    h = conditional_differential_entropy_from_k(K, A=[0], C=[1])
    assert not h.is_complex()
    assert h.dtype == torch.float64
    assert h.ndim == 0
    assert h.device == K[(0, 0)].device


# ============================================================
# Test 9: Validation — non-empty A and A-C disjointness.
# ============================================================


def test_empty_A_rejected():
    K = _build_mac_K(d=2, seed=810)
    with pytest.raises(ValueError, match="non-empty"):
        conditional_differential_entropy_from_k(K, A=[], C=[1])


def test_overlapping_A_C_rejected():
    K = _build_mac_K(d=2, seed=811)
    with pytest.raises(ValueError, match="disjoint"):
        conditional_differential_entropy_from_k(K, A=[0, 1], C=[1])


# ============================================================
# Analytic ground-truth tests.
#
# These compare against textbook closed forms whose reference value is a
# pure scalar expression (math.log only) — no matrix routine touches the
# reference side, so they are the strongest correctness anchor.
#   h(CN(0, sigma^2))             = log(pi e sigma^2)               (scalar)
#   h(CN(0, diag(sigma_i^2)))     = sum_i log(pi e sigma_i^2)       (diagonal)
#   AWGN output  Y = g X + Z:     h(Y)   = log(pi e (|g|^2 P + N))
#                                 h(Y|X) = log(pi e N)              (= h(Z))
#                                 h(Y) - h(Y|X) = log(1 + |g|^2 P / N)
# ============================================================


def _scalar_dag_K(sigma2: float):
    """1 scalar root with variance sigma2 (node 1 is an unused sink)."""
    return compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents={1: [0]},
        edge_mats={(1, 0): torch.eye(1, dtype=DTYPE)},
        root_covs={0: torch.tensor([[sigma2]], dtype=DTYPE)},
        noise_covs={1: torch.eye(1, dtype=DTYPE)},
    )


def test_scalar_gaussian_entropy_analytic():
    """h(X) = log(pi e sigma^2) for a scalar CN(0, sigma^2)."""
    for sigma2 in [0.25, 1.0, 2.0, 7.3]:
        K = _scalar_dag_K(sigma2)
        h_lib = conditional_differential_entropy_from_k(K, A=[0])
        h_exact = math.log(math.pi * math.e * sigma2)
        assert math.isclose(h_lib.item(), h_exact, rel_tol=0, abs_tol=1e-12), (
            f"sigma2={sigma2}: got {h_lib.item()}, expected {h_exact}"
        )


def test_diagonal_gaussian_entropy_analytic():
    """h(X) = sum_i log(pi e sigma_i^2) for a diagonal-covariance Gaussian."""
    variances = [0.5, 1.0, 3.0, 4.2]
    d = len(variances)
    Sigma = torch.diag(torch.tensor(variances, dtype=torch.float64)).to(DTYPE)
    K = compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents={1: [0]},
        edge_mats={(1, 0): torch.eye(d, dtype=DTYPE)},
        root_covs={0: Sigma},
        noise_covs={1: torch.eye(d, dtype=DTYPE)},
    )
    h_lib = conditional_differential_entropy_from_k(K, A=[0])
    h_exact = sum(math.log(math.pi * math.e * v) for v in variances)
    assert math.isclose(h_lib.item(), h_exact, rel_tol=0, abs_tol=1e-12)


def test_scalar_awgn_output_and_conditional_entropy_analytic():
    """Scalar AWGN link Y = g X + Z, X~CN(0,P), Z~CN(0,N):
        h(Y)   = log(pi e (|g|^2 P + N)),
        h(Y|X) = log(pi e N),
        h(Y) - h(Y|X) = log(1 + |g|^2 P / N)   (the classical capacity).
    """
    P, N = 1.7, 0.6
    g = complex(0.8, -0.5)
    K = compute_k_blocks_multiroot(
        num_nodes=2, roots=[0], parents={1: [0]},
        edge_mats={(1, 0): torch.tensor([[g]], dtype=DTYPE)},
        root_covs={0: torch.tensor([[P]], dtype=DTYPE)},
        noise_covs={1: torch.tensor([[N]], dtype=DTYPE)},
    )
    h_Y = conditional_differential_entropy_from_k(K, A=[1])
    h_Y_given_X = conditional_differential_entropy_from_k(K, A=[1], C=[0])

    gain = abs(g) ** 2
    h_Y_exact = math.log(math.pi * math.e * (gain * P + N))
    h_Y_given_X_exact = math.log(math.pi * math.e * N)
    cap_exact = math.log(1.0 + gain * P / N)

    assert math.isclose(h_Y.item(), h_Y_exact, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        h_Y_given_X.item(), h_Y_given_X_exact, rel_tol=0, abs_tol=1e-12
    )
    assert math.isclose(
        (h_Y - h_Y_given_X).item(), cap_exact, rel_tol=0, abs_tol=1e-12
    )


def test_scalar_mac_conditional_entropies_analytic():
    """2-user scalar MAC Y = g1 X1 + g2 X2 + Z, X_k~CN(0,P_k), Z~CN(0,N):
        h(Y | X1, X2) = log(pi e N),
        h(Y | X2)     = log(pi e (|g1|^2 P1 + N)),
        h(Y)          = log(pi e (|g1|^2 P1 + |g2|^2 P2 + N)).
    The MAC facet I(X1; Y | X2) = h(Y|X2) - h(Y|X1,X2) then recovers the
    classical log(1 + |g1|^2 P1 / N)."""
    P1, P2, N = 1.3, 0.9, 0.5
    g1, g2 = complex(0.7, 0.2), complex(-0.4, 0.6)
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): torch.tensor([[g1]], dtype=DTYPE),
                   (2, 1): torch.tensor([[g2]], dtype=DTYPE)},
        root_covs={0: torch.tensor([[P1]], dtype=DTYPE),
                   1: torch.tensor([[P2]], dtype=DTYPE)},
        noise_covs={2: torch.tensor([[N]], dtype=DTYPE)},
    )
    s1, s2 = abs(g1) ** 2 * P1, abs(g2) ** 2 * P2

    h_Y = conditional_differential_entropy_from_k(K, A=[2])
    h_Y_given_X2 = conditional_differential_entropy_from_k(K, A=[2], C=[1])
    h_Y_given_both = conditional_differential_entropy_from_k(K, A=[2], C=[0, 1])

    assert math.isclose(
        h_Y.item(), math.log(math.pi * math.e * (s1 + s2 + N)),
        rel_tol=0, abs_tol=1e-12,
    )
    assert math.isclose(
        h_Y_given_X2.item(), math.log(math.pi * math.e * (s1 + N)),
        rel_tol=0, abs_tol=1e-12,
    )
    assert math.isclose(
        h_Y_given_both.item(), math.log(math.pi * math.e * N),
        rel_tol=0, abs_tol=1e-12,
    )
    # Facet I(X1; Y | X2) via entropies == classical capacity.
    I1 = h_Y_given_X2 - h_Y_given_both
    assert math.isclose(
        I1.item(), math.log(1.0 + s1 / N), rel_tol=0, abs_tol=1e-12
    )
