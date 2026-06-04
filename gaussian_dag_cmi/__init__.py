"""Gaussian-DAG-CMI: conditional mutual information and projected gradient
descent for multi-terminal linear Gaussian DAGs.

Sister library to `gaussian-dag`. Extends the parent's single-root
K-recursion and single-pair mutual information to the multi-root case and
to conditional MI on arbitrary disjoint subsets, and adds a projected
gradient descent loop `pga_descent` mirroring the parent's `pga_ascent`.

Numerical primitives (`logdet_hpd`, `pga_ascent`, `get_K`, `hermitianize`,
`project_frobenius_ball`, `project_total_power`) live in the parent library
and are imported from there rather than duplicated.
"""

from gaussian_dag_cmi.information import conditional_mutual_information_from_k
from gaussian_dag_cmi.krecursion import compute_k_blocks_multiroot
from gaussian_dag_cmi.optimize import pga_descent
from gaussian_dag_cmi.rate_region import Summand, evaluate_rate_functions

__all__ = [
    "Summand",
    "compute_k_blocks_multiroot",
    "conditional_mutual_information_from_k",
    "evaluate_rate_functions",
    "pga_descent",
]
