"""Closed-form value comparisons for the 2-user Gaussian MAC.

These tests verify that the multi-root K-recursion + conditional MI pipeline
returns numerically identical values to classical closed-form expressions
for the 2-user MAC pentagon facets:

  Scalar MAC (d = 1):
      I(X1; Y | X2) = log(1 + |h1|^2 P1 / sigma^2)
      I(X2; Y | X1) = log(1 + |h2|^2 P2 / sigma^2)
      I(X1, X2; Y) = log(1 + (|h1|^2 P1 + |h2|^2 P2) / sigma^2)

  MIMO MAC (general d):
      I(X1; Y | X2) = log det(I + (1/sigma^2) H1 F1 F1^H H1^H)
      I(X2; Y | X1) = log det(I + (1/sigma^2) H2 F2 F2^H H2^H)
      I(X1, X2; Y) = log det(I + (1/sigma^2) (H1 F1 F1^H H1^H
                                            + H2 F2 F2^H H2^H))

The MAC pentagon's three CMI's are the canonical building block of every
downstream rate-region computation, so an end-to-end agreement with the
classical formulas is the strongest correctness statement we can make about
the library short of a full proof.
"""

from __future__ import annotations

import math

import torch

from cmi_dag import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
)

DTYPE = torch.complex128


def _randn_complex(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(*shape, dtype=torch.float64, generator=g)
    im = torch.randn(*shape, dtype=torch.float64, generator=g)
    return torch.complex(re, im)


def _logdet_hpd(A: torch.Tensor) -> torch.Tensor:
    """Hermitian-positive-definite log-det via Cholesky (reference implementation)."""
    A = 0.5 * (A + A.mH)
    L = torch.linalg.cholesky(A)
    return 2.0 * torch.log(torch.diagonal(L).real).sum()


def _mac_K(F1, F2, H1, H2, sigma2):
    """Run the multi-root K-recursion for a 2-user MAC with the given precoders
    and channels (un-batched, complex128)."""
    d1 = F1.shape[-1]
    d2 = F2.shape[-1]
    d_y = H1.shape[-2]
    return compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0, 1],
        parents={2: [0, 1]},
        edge_mats={(2, 0): H1 @ F1, (2, 1): H2 @ F2},
        root_covs={0: torch.eye(d1, dtype=DTYPE),
                   1: torch.eye(d2, dtype=DTYPE)},
        noise_covs={2: sigma2 * torch.eye(d_y, dtype=DTYPE)},
    )


# ============================================================
# Test 1a: Scalar MAC, I(X1; Y | X2)  vs  log(1 + |h1|^2 P1 / sigma^2).
# ============================================================


def test_scalar_mac_I1_closed_form():
    """For d = 1, F_k = sqrt(P_k), inputs CN(0,1): I_lib == log(1 + |h1|^2 P1 / sigma^2)."""
    sigma2 = 0.7
    P1, P2 = 1.3, 0.4
    h1 = complex(0.8, -0.5)
    h2 = complex(-0.2, 0.9)

    F1 = torch.tensor([[math.sqrt(P1)]], dtype=DTYPE)
    F2 = torch.tensor([[math.sqrt(P2)]], dtype=DTYPE)
    H1 = torch.tensor([[h1]], dtype=DTYPE)
    H2 = torch.tensor([[h2]], dtype=DTYPE)

    K = _mac_K(F1, F2, H1, H2, sigma2)
    I1_lib = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    I1_ref = torch.tensor(math.log(1.0 + abs(h1) ** 2 * P1 / sigma2),
                          dtype=torch.float64)

    assert torch.allclose(I1_lib, I1_ref, atol=1e-12), (
        f"I_lib = {I1_lib.item():.6e}, I_ref = {I1_ref.item():.6e}"
    )


# ============================================================
# Test 1b: Scalar MAC, I(X2; Y | X1)  vs  log(1 + |h2|^2 P2 / sigma^2).
# ============================================================


def test_scalar_mac_I2_closed_form():
    sigma2 = 0.7
    P1, P2 = 1.3, 0.4
    h1 = complex(0.8, -0.5)
    h2 = complex(-0.2, 0.9)

    F1 = torch.tensor([[math.sqrt(P1)]], dtype=DTYPE)
    F2 = torch.tensor([[math.sqrt(P2)]], dtype=DTYPE)
    H1 = torch.tensor([[h1]], dtype=DTYPE)
    H2 = torch.tensor([[h2]], dtype=DTYPE)

    K = _mac_K(F1, F2, H1, H2, sigma2)
    I2_lib = conditional_mutual_information_from_k(K, A=[1], B=[2], C=[0])
    I2_ref = torch.tensor(math.log(1.0 + abs(h2) ** 2 * P2 / sigma2),
                          dtype=torch.float64)

    assert torch.allclose(I2_lib, I2_ref, atol=1e-12)


# ============================================================
# Test 1c: Scalar MAC, I(X1, X2; Y)  vs  log(1 + sum_k |h_k|^2 P_k / sigma^2).
# ============================================================


def test_scalar_mac_I12_closed_form():
    sigma2 = 0.7
    P1, P2 = 1.3, 0.4
    h1 = complex(0.8, -0.5)
    h2 = complex(-0.2, 0.9)

    F1 = torch.tensor([[math.sqrt(P1)]], dtype=DTYPE)
    F2 = torch.tensor([[math.sqrt(P2)]], dtype=DTYPE)
    H1 = torch.tensor([[h1]], dtype=DTYPE)
    H2 = torch.tensor([[h2]], dtype=DTYPE)

    K = _mac_K(F1, F2, H1, H2, sigma2)
    I12_lib = conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[])
    snr_sum = (abs(h1) ** 2 * P1 + abs(h2) ** 2 * P2) / sigma2
    I12_ref = torch.tensor(math.log(1.0 + snr_sum), dtype=torch.float64)

    assert torch.allclose(I12_lib, I12_ref, atol=1e-12)


# ============================================================
# Test 2a: MIMO MAC, I(X1; Y | X2)  vs
#          log det(I_dy + (1/sigma^2) H1 F1 F1^H H1^H).
# ============================================================


def test_mimo_mac_I1_closed_form():
    d1, d2, d_y = 3, 2, 4
    sigma2 = 0.6
    F1 = _randn_complex(d1, d1, seed=1).to(DTYPE)
    F2 = _randn_complex(d2, d2, seed=2).to(DTYPE)
    H1 = _randn_complex(d_y, d1, seed=3).to(DTYPE)
    H2 = _randn_complex(d_y, d2, seed=4).to(DTYPE)

    K = _mac_K(F1, F2, H1, H2, sigma2)
    I1_lib = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])

    eye_y = torch.eye(d_y, dtype=DTYPE)
    M = H1 @ F1 @ F1.mH @ H1.mH
    I1_ref = _logdet_hpd(eye_y + M / sigma2)

    assert torch.allclose(I1_lib, I1_ref, atol=1e-10), (
        f"I_lib = {I1_lib.item():.6e}, I_ref = {I1_ref.item():.6e}, "
        f"diff = {(I1_lib - I1_ref).abs().item():.3e}"
    )


# ============================================================
# Test 2b: MIMO MAC, I(X2; Y | X1)  --  symmetric to 2a.
# ============================================================


def test_mimo_mac_I2_closed_form():
    d1, d2, d_y = 3, 2, 4
    sigma2 = 0.6
    F1 = _randn_complex(d1, d1, seed=10).to(DTYPE)
    F2 = _randn_complex(d2, d2, seed=11).to(DTYPE)
    H1 = _randn_complex(d_y, d1, seed=12).to(DTYPE)
    H2 = _randn_complex(d_y, d2, seed=13).to(DTYPE)

    K = _mac_K(F1, F2, H1, H2, sigma2)
    I2_lib = conditional_mutual_information_from_k(K, A=[1], B=[2], C=[0])

    eye_y = torch.eye(d_y, dtype=DTYPE)
    M = H2 @ F2 @ F2.mH @ H2.mH
    I2_ref = _logdet_hpd(eye_y + M / sigma2)

    assert torch.allclose(I2_lib, I2_ref, atol=1e-10)


# ============================================================
# Test 2c: MIMO MAC, I(X1, X2; Y) vs
#          log det(I_dy + (1/sigma^2) (H1 F1 F1^H H1^H + H2 F2 F2^H H2^H)).
# ============================================================


def test_mimo_mac_I12_closed_form():
    d1, d2, d_y = 3, 2, 4
    sigma2 = 0.6
    F1 = _randn_complex(d1, d1, seed=20).to(DTYPE)
    F2 = _randn_complex(d2, d2, seed=21).to(DTYPE)
    H1 = _randn_complex(d_y, d1, seed=22).to(DTYPE)
    H2 = _randn_complex(d_y, d2, seed=23).to(DTYPE)

    K = _mac_K(F1, F2, H1, H2, sigma2)
    I12_lib = conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[])

    eye_y = torch.eye(d_y, dtype=DTYPE)
    M = H1 @ F1 @ F1.mH @ H1.mH + H2 @ F2 @ F2.mH @ H2.mH
    I12_ref = _logdet_hpd(eye_y + M / sigma2)

    assert torch.allclose(I12_lib, I12_ref, atol=1e-10)
