"""Projected gradient DESCENT for cost-type objectives on linear Gaussian DAGs.

Minimization analog of `gaussian_dag.pga_ascent`. Use `pga_descent` for
objectives that are to be *minimized* (e.g., outage probability, equalization
error), and the parent's `pga_ascent` for objectives that are to be
*maximized* (e.g., mutual information, throughput). The two functions have
the same signature, return-type, and history convention; only the optimization
direction is reversed.

Implementation: the closure is internally negated and forwarded to the
parent's `pga_ascent`; the returned history is sign-flipped so that values
are reported in the true sign of the user's objective (i.e., a decreasing
history corresponds to successful minimization).

The closure and projector contracts are identical to the parent's:
- Closure: a no-argument callable returning a real scalar tensor,
  differentiable through `params`.
- Projector: either an in-place mutator returning `None`, or a functional
  projector returning a sequence of new tensors (one per parameter).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from gaussian_dag.optimize import pga_ascent


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
