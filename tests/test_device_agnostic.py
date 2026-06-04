"""Device-agnostic smoke tests for the full pipeline.

The new library is device-agnostic: every internally allocated tensor
inherits `dtype` and `device` from an input tensor. These tests verify that
the full pipeline (`compute_k_blocks_multiroot` ->
`conditional_mutual_information_from_k` -> `pga_descent`) runs end-to-end on
every device available on the current machine, and that the numerical
results agree across devices.

CUDA and MPS tests are skipped when the corresponding backend is unavailable.
"""

from __future__ import annotations

import pytest
import torch

from gaussian_dag.projections import project_frobenius_ball
from gaussian_dag_cmi import (
    compute_k_blocks_multiroot,
    conditional_mutual_information_from_k,
    pga_descent,
)

DTYPE = torch.complex128


def _available_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS does not support complex128, so skip when we use complex128.
        # The user may opt in to complex64 via a future kwarg.
        pass
    return devices


def _build_mac_problem(device: torch.device, *, seed: int = 0):
    """Construct a deterministic 2-user MAC problem on the given device."""
    g = torch.Generator().manual_seed(seed)
    d = 2
    H1 = torch.complex(
        torch.randn(d, d, dtype=torch.float64, generator=g),
        torch.randn(d, d, dtype=torch.float64, generator=g),
    ).to(DTYPE).to(device)
    H2 = torch.complex(
        torch.randn(d, d, dtype=torch.float64, generator=g),
        torch.randn(d, d, dtype=torch.float64, generator=g),
    ).to(DTYPE).to(device)
    F1 = (0.5 * torch.eye(d, dtype=DTYPE, device=device)).requires_grad_(True)
    F2 = (0.5 * torch.eye(d, dtype=DTYPE, device=device)).requires_grad_(True)
    Sigma_0 = torch.eye(d, dtype=DTYPE, device=device)
    Sigma_1 = torch.eye(d, dtype=DTYPE, device=device)
    Sigma_Y = torch.eye(d, dtype=DTYPE, device=device)
    return H1, H2, F1, F2, Sigma_0, Sigma_1, Sigma_Y


def _evaluate_pentagon(H1, H2, F1, F2, Sigma_0, Sigma_1, Sigma_Y):
    K = compute_k_blocks_multiroot(
        num_nodes=3, roots=[0, 1], parents={2: [0, 1]},
        edge_mats={(2, 0): H1 @ F1, (2, 1): H2 @ F2},
        root_covs={0: Sigma_0, 1: Sigma_1},
        noise_covs={2: Sigma_Y},
    )
    return (
        K,
        conditional_mutual_information_from_k(K, A=[0],    B=[2], C=[1]),
        conditional_mutual_information_from_k(K, A=[1],    B=[2], C=[0]),
        conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[]),
    )


# ============================================================
# Test 1: every available device runs the full pipeline; output tensors
#         are placed on the requested device.
# ============================================================


@pytest.mark.parametrize("device", _available_devices(), ids=lambda d: str(d))
def test_pipeline_runs_on_device(device: torch.device):
    H1, H2, F1, F2, Sigma_0, Sigma_1, Sigma_Y = _build_mac_problem(device)
    K, I1, I2, I_sigma = _evaluate_pentagon(H1, H2, F1, F2, Sigma_0, Sigma_1, Sigma_Y)

    for key in K:
        assert K[key].device.type == device.type, (
            f"K{key} on {K[key].device}, expected {device}"
        )
    for I in (I1, I2, I_sigma):
        assert I.device.type == device.type

    # All three pentagon MIs must be finite and non-negative.
    for name, I in (("I1", I1), ("I2", I2), ("I_sigma", I_sigma)):
        v = I.item()
        assert torch.isfinite(torch.tensor(v)), f"{name} not finite: {v}"
        assert v >= -1e-12, f"{name} negative: {v}"


# ============================================================
# Test 2: CUDA agrees with CPU on every K-block and every pentagon MI.
# Skipped when CUDA is not available.
# ============================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_cuda_matches_cpu():
    cpu = torch.device("cpu")
    gpu = torch.device("cuda")

    prob_cpu = _build_mac_problem(cpu, seed=42)
    prob_gpu = _build_mac_problem(gpu, seed=42)

    K_cpu, I1_cpu, I2_cpu, IS_cpu = _evaluate_pentagon(*prob_cpu)
    K_gpu, I1_gpu, I2_gpu, IS_gpu = _evaluate_pentagon(*prob_gpu)

    # K-blocks: same keys, agreement within complex128 jitter.
    assert set(K_cpu.keys()) == set(K_gpu.keys())
    for key in K_cpu:
        diff = (K_cpu[key] - K_gpu[key].cpu()).abs().max().item()
        assert diff < 1e-10, f"K{key} CPU vs CUDA differs by {diff}"

    # Pentagon MIs.
    assert abs(I1_cpu.item() - I1_gpu.item()) < 1e-10
    assert abs(I2_cpu.item() - I2_gpu.item()) < 1e-10
    assert abs(IS_cpu.item() - IS_gpu.item()) < 1e-10


# ============================================================
# Test 3: pga_descent runs on every available device and produces a
#         monotone non-increasing history on a strongly convex problem.
# ============================================================


@pytest.mark.parametrize("device", _available_devices(), ids=lambda d: str(d))
def test_pga_descent_on_device(device: torch.device):
    d = 2
    F_target = torch.eye(d, dtype=DTYPE, device=device)
    F = torch.zeros(d, d, dtype=DTYPE, device=device, requires_grad=True)

    def closure() -> torch.Tensor:
        diff = F - F_target
        return torch.real(torch.sum(diff.conj() * diff))

    def projector(params):
        return [project_frobenius_ball(params[0], P=10.0)]

    history = pga_descent(
        closure, [F], step_size=0.1, num_iters=50, projector=projector
    )

    assert F.device.type == device.type
    # Monotone non-increasing (allow tiny FP noise).
    for prev, curr in zip(history, history[1:]):
        assert curr <= prev + 1e-10
    assert history[-1] < 1e-6
