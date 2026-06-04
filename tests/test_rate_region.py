"""Unit tests for gaussian_dag_cmi.rate_region.evaluate_rate_functions."""

from __future__ import annotations

import pytest
import torch

from gaussian_dag_cmi.information import conditional_mutual_information_from_k
from gaussian_dag_cmi.krecursion import compute_k_blocks_multiroot
from gaussian_dag_cmi.rate_region import evaluate_rate_functions

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


# ============================================================
# Test 1: MAC pentagon as a 3-inequality family (each N_T = 1).
# ============================================================


def test_mac_pentagon_evaluation():
    K = _build_mac_K(d=2, seed=700)
    inequalities = [
        [(1.0, [0],    [2], [1])],   # T={1}:    I(X1; Y | X2)
        [(1.0, [1],    [2], [0])],   # T={2}:    I(X2; Y | X1)
        [(1.0, [0, 1], [2], [])],    # T={1,2}:  I(X1, X2; Y)
    ]
    f_T_list = evaluate_rate_functions(K, inequalities)

    assert len(f_T_list) == 3
    # Each must agree with the direct call.
    expected = [
        conditional_mutual_information_from_k(K, A=[0],    B=[2], C=[1]),
        conditional_mutual_information_from_k(K, A=[1],    B=[2], C=[0]),
        conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[]),
    ]
    for f, e in zip(f_T_list, expected):
        assert torch.allclose(f, e, atol=1e-12)


# ============================================================
# Test 2: Linear combinations with multiple summands (N_T >= 2),
#         including a negative coefficient (HK / DF-style differences).
# ============================================================


def test_linear_combination_with_negative_alpha():
    K = _build_mac_K(d=2, seed=800)
    inequalities = [
        [(1.0, [0, 1], [2], []),
         (-1.0, [1],    [2], [])],   # I(X1, X2; Y) - I(X2; Y)
    ]
    f_T_list = evaluate_rate_functions(K, inequalities)

    # By the chain rule, this equals I(X1; Y | X2).
    expected = conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1])
    assert torch.allclose(f_T_list[0], expected, atol=1e-10)


# ============================================================
# Test 3: Empty summand sequence raises ValueError.
# ============================================================


def test_empty_summands_raises():
    K = _build_mac_K(d=2, seed=900)
    with pytest.raises(ValueError, match="at least one summand"):
        evaluate_rate_functions(K, inequalities=[[]])


# ============================================================
# Test 4: Differentiability through K (via edge_mats).
# ============================================================


def test_gradient_through_rate_function():
    d = 2
    F = _randn_complex(d, d, seed=1000).requires_grad_(True)
    H = _randn_complex(d, d, seed=1001)
    A0 = H @ F
    A1 = torch.eye(d, dtype=DTYPE)

    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): A0, (2, 1): A1},
        root_covs={0: torch.eye(d, dtype=DTYPE),
                   1: torch.eye(d, dtype=DTYPE)},
        noise_covs={2: torch.eye(d, dtype=DTYPE)},
    )
    inequalities = [[(1.0, [0], [2], [1])]]
    f_T = evaluate_rate_functions(K, inequalities)[0]
    f_T.backward()
    assert F.grad is not None
    assert torch.isfinite(F.grad).all()
