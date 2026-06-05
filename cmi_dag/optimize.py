"""Projected gradient ASCENT / DESCENT for objectives on linear Gaussian DAGs.

`pga_ascent` is the topology-agnostic constant-step-size projected gradient
ascent loop (vendored from `gaussian_dag.optimize`; the same numerical
primitive, kept here so this library is fully self-contained with no
`gaussian-dag` runtime dependency).

`pga_descent` is its minimization analog. Use `pga_descent` for objectives
that are to be *minimized* (e.g., outage probability, equalization error),
and `pga_ascent` for objectives that are to be *maximized* (e.g., mutual
information, throughput). The two functions have the same signature,
return-type, and history convention; only the optimization direction is
reversed.

Implementation: `pga_descent` internally negates the closure and forwards
to `pga_ascent`; the returned history is sign-flipped so that values are
reported in the true sign of the user's objective (i.e., a decreasing
history corresponds to successful minimization).

The closure and projector contracts are identical for both:
- Closure: a no-argument callable returning a real scalar tensor,
  differentiable through `params`.
- Projector: either an in-place mutator returning `None`, or a functional
  projector returning a sequence of new tensors (one per parameter).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


def pga_ascent(
    compute_mi: Callable[[], torch.Tensor],
    params: list[torch.Tensor],
    *,
    step_size: float,
    num_iters: int,
    projector: Callable[[list[torch.Tensor]], None | Sequence[torch.Tensor]] | None = None,
) -> list[float]:
    """Run projected gradient ASCENT on a scalar objective.

    At each iteration t = 0, 1, ..., num_iters - 1:
        1. Zero out any existing gradients in `params`.
        2. Compute `I = compute_mi()` and call `I.backward()`.
           Record `I.item()` in the history list.
        3. Inside `torch.no_grad()`:
           a. Update each `p in params` via `p.add_(step_size * p.grad)`.
           b. If `projector` is provided, call `projector(params)`. The
              projector may either mutate `params` in place and return
              `None`, or return a sequence of new tensors (one per
              parameter) which `pga_ascent` will copy into place. Mixing
              the two is not supported within one call.

    Args:
        compute_mi: Closure that, given the current state of `params`,
            constructs the autograd graph and returns the scalar objective.
        params: List of leaf tensors with `requires_grad=True`.
        step_size: Constant positive step size.
        num_iters: Number of PGA iterations (must be > 0).
        projector: Optional callable taking `params`. May either mutate in
            place (returning `None`) or return a sequence of new tensors
            (one per parameter), in which case the returned tensors are
            copied into the parameters by `pga_ascent`.

    Returns:
        history: List of length `num_iters`, where history[t] = I.item()
            recorded immediately after the forward pass of iteration t
            (i.e., the objective evaluated at the *pre-update* parameter
            values for iteration t).
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be positive, got {num_iters}")
    for p in params:
        if not p.requires_grad:
            raise ValueError("All entries of `params` must have requires_grad=True.")

    history: list[float] = []
    for _ in range(num_iters):
        # 1. Zero out previous gradients (allowed to be None on first iteration).
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        # 2. Forward + backward.
        I = compute_mi()
        I.backward()
        history.append(I.item())
        # 3. Update and project.
        with torch.no_grad():
            for idx, p in enumerate(params):
                if p.grad is None:
                    raise RuntimeError(
                        f"params[{idx}] received no gradient after backward(): "
                        "the parameter has requires_grad=True but does not "
                        "participate in the autograd graph produced by "
                        "compute_mi(). Common causes: (a) the parameter is "
                        "declared but never used in the closure; (b) the "
                        "closure rebinds the parameter to a new tensor "
                        "(e.g. via `F = F.detach()` or in-place arithmetic "
                        "outside torch.no_grad); (c) a typo in a closure-"
                        "captured variable name. Verify that the parameter "
                        "is referenced inside compute_mi() and that the "
                        "returned objective tensor depends on it."
                    )
                p.add_(step_size * p.grad)
            if projector is not None:
                out = projector(params)
                if out is not None:
                    # Functional projector: copy each returned tensor into place.
                    if len(out) != len(params):
                        raise ValueError(
                            f"projector returned {len(out)} tensors, expected "
                            f"{len(params)}."
                        )
                    for p, q in zip(params, out):
                        p.copy_(q)
    return history


def pga_descent(
    closure: Callable[[], torch.Tensor],
    params: list[torch.Tensor],
    *,
    step_size: float,
    num_iters: int,
    projector: Callable[[list[torch.Tensor]], None | Sequence[torch.Tensor]] | None = None,
) -> list[float]:
    """Run projected gradient DESCENT on a cost-type objective.

    Minimization analog of `gaussian_dag.pga_ascent`: the function has the
    identical signature and history convention, but descends the objective
    rather than ascending it. Internally negates the closure and forwards
    to `pga_ascent`, then flips the returned history sign so values are
    reported in the true sign of the user's objective.

    Args:
        closure: Closure that, given the current state of `params`,
            constructs the autograd graph and returns the scalar cost
            tensor (the quantity to be minimized).
        params: List of leaf tensors with `requires_grad=True`.
        step_size: Constant positive step size.
        num_iters: Number of PGD iterations (must be > 0).
        projector: Optional callable taking `params`. May either mutate in
            place (returning `None`) or return a sequence of new tensors
            (one per parameter); see `gaussian_dag.pga_ascent` for the full
            contract.

    Returns:
        history: List of length `num_iters`, where history[t] is the value
            of the *user-facing* cost at iteration t (in the true sign of
            the objective, i.e., monotonically decreasing for a successful
            descent). The value is recorded at the iteration's pre-update
            parameter state, inheriting the convention of the parent's
            `pga_ascent`.
    """
    def negated_closure() -> torch.Tensor:
        return -closure()

    flipped_history = pga_ascent(
        negated_closure,
        params,
        step_size=step_size,
        num_iters=num_iters,
        projector=projector,
    )
    return [-h for h in flipped_history]
