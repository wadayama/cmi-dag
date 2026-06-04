"""Unit tests for cmi_dag.optimize.pga_descent."""

from __future__ import annotations

import pytest
import torch

from gaussian_dag.optimize import pga_ascent
from cmi_dag.optimize import pga_descent

DTYPE = torch.complex128


# ============================================================
# Test 1: Descending a quadratic cost converges toward the minimum.
# Cost(F) = || F - F_target ||_F^2, minimum at F = F_target.
# ============================================================


def test_descent_on_quadratic_cost():
    d = 3
    torch.manual_seed(0)
    F_target = torch.randn(d, d, dtype=DTYPE)

    F = torch.zeros(d, d, dtype=DTYPE, requires_grad=True)

    def closure() -> torch.Tensor:
        diff = F - F_target
        return torch.real(torch.sum(diff.conj() * diff))

    history = pga_descent(closure, [F], step_size=0.1, num_iters=100)

    assert len(history) == 100
    # Monotone descent (allow some noise tolerance).
    assert history[-1] < history[0]
    # Final cost should be near zero.
    assert history[-1] < 1e-6


# ============================================================
# Test 2: pga_descent on -f matches pga_ascent on f.
# Sign-flip equivalence: minimizing -f and maximizing f give the same iterates.
# ============================================================


def test_descent_negated_matches_ascent():
    d = 2
    torch.manual_seed(0)
    A = torch.randn(d, d, dtype=DTYPE)
    Q = A @ A.mH + torch.eye(d, dtype=DTYPE)  # PSD

    F_a = torch.eye(d, dtype=DTYPE, requires_grad=True)
    F_d = torch.eye(d, dtype=DTYPE, requires_grad=True)

    def f_ascent() -> torch.Tensor:
        return torch.real(torch.trace(F_a.mH @ Q @ F_a))

    def f_descent_neg() -> torch.Tensor:
        return -torch.real(torch.trace(F_d.mH @ Q @ F_d))

    hist_asc = pga_ascent(f_ascent, [F_a], step_size=0.01, num_iters=20)
    hist_desc = pga_descent(f_descent_neg, [F_d], step_size=0.01, num_iters=20)

    # pga_descent returns the user-facing history (the negated objective).
    # The two iterates should track each other exactly.
    for ha, hd in zip(hist_asc, hist_desc):
        # ha is f, hd is -f (the user-facing cost). So hd == -ha.
        assert abs(ha + hd) < 1e-10
    # And the parameters should agree.
    assert torch.allclose(F_a, F_d, atol=1e-10)


# ============================================================
# Test 3: history reports cost in true sign (decreasing).
# ============================================================


def test_history_in_true_sign():
    d = 2
    F = torch.eye(d, dtype=DTYPE, requires_grad=True)

    def closure() -> torch.Tensor:
        return torch.real(torch.sum(F.conj() * F))  # ||F||_F^2 >= 0

    history = pga_descent(closure, [F], step_size=0.1, num_iters=10)

    # All values must be non-negative (the cost is non-negative).
    for v in history:
        assert v >= -1e-12
    # Monotone non-increasing.
    for prev, curr in zip(history, history[1:]):
        assert curr <= prev + 1e-12


# ============================================================
# Test 4: projector contract — functional projector projects after each step.
# ============================================================


def test_descent_with_functional_projector():
    d = 2
    from gaussian_dag.projections import project_frobenius_ball

    F_target = 3.0 * torch.eye(d, dtype=DTYPE)
    F = torch.zeros(d, dtype=DTYPE).repeat(d, 1).requires_grad_(True)

    def closure() -> torch.Tensor:
        diff = F - F_target
        return torch.real(torch.sum(diff.conj() * diff))

    # Constrain ||F||_F^2 <= 1.
    def projector(params):
        return [project_frobenius_ball(params[0], P=1.0)]

    pga_descent(closure, [F], step_size=0.1, num_iters=200, projector=projector)
    # F must lie within the Frobenius ball after the loop.
    assert torch.linalg.norm(F).item() ** 2 <= 1.0 + 1e-8


# ============================================================
# Test 5: invalid inputs raise (via parent pga_ascent's validation).
# ============================================================


def test_invalid_step_size():
    F = torch.eye(2, dtype=DTYPE, requires_grad=True)
    with pytest.raises(ValueError, match="step_size"):
        pga_descent(lambda: torch.real(torch.sum(F.conj() * F)), [F],
                    step_size=0.0, num_iters=1)


def test_invalid_num_iters():
    F = torch.eye(2, dtype=DTYPE, requires_grad=True)
    with pytest.raises(ValueError, match="num_iters"):
        pga_descent(lambda: torch.real(torch.sum(F.conj() * F)), [F],
                    step_size=0.1, num_iters=0)
