"""cmi-dag: conditional mutual information and projected gradient descent
for multi-terminal linear Gaussian DAGs.

Sister library to `gaussian-dag`. Generalizes single-root K-recursion and
single-pair mutual information to the multi-root case and to conditional MI
on arbitrary disjoint subsets, and adds a projected gradient descent loop
`pga_descent` alongside `pga_ascent`.

This library is fully self-contained: the multi-root K-recursion is its own
contribution, and the generic numerical primitives (`logdet_hpd`,
`pga_ascent`, `get_K`, `hermitianize`, `project_frobenius_ball`,
`project_total_power`) are vendored here so cmi-dag has no `gaussian-dag`
runtime dependency. These primitives are byte-identical to those in
`gaussian-dag`, which remains the single-root reference implementation of its
own paper.
"""

from cmi_dag.information import (
    conditional_differential_entropy_from_k,
    conditional_mutual_information_from_k,
    logdet_hpd,
)
from cmi_dag.krecursion import (
    compute_effective_channel,
    compute_k_blocks_multiroot,
    get_K,
    hermitianize,
)
from cmi_dag.optimize import pga_ascent, pga_descent
from cmi_dag.projections import project_frobenius_ball, project_total_power
from cmi_dag.rate_region import Summand, evaluate_rate_functions

__all__ = [
    "Summand",
    "compute_effective_channel",
    "compute_k_blocks_multiroot",
    "conditional_differential_entropy_from_k",
    "conditional_mutual_information_from_k",
    "evaluate_rate_functions",
    "get_K",
    "hermitianize",
    "logdet_hpd",
    "pga_ascent",
    "pga_descent",
    "project_frobenius_ball",
    "project_total_power",
]
