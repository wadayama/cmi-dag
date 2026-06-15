"""Named-node DAG builder: a declarative front-end over the multi-root core.

This module is a *pure, backward-compatible addition* to the library. It adds
no new behavior to the numerical core; it only provides a convenience surface
that *lowers* a named-node DAG declaration to the existing functional API
(``compute_k_blocks_multiroot`` -> ``conditional_mutual_information_from_k`` /
``get_K``).

Worked example (a 2-user MAC ``X1, X2 -> Y``)::

    dag = GaussianDAG()
    dag.add_source("X1", cov=Sigma1)            # multiple roots are allowed
    dag.add_source("X2", cov=Sigma2)
    dag.add_node("Y", parents={"X1": H1, "X2": H2}, noise=N_Y)

    I1  = dag.cmi(A=["X1"], B=["Y"], C=["X2"])  # I(X1; Y | X2) in nats
    I12 = dag.cmi(A=["X1", "X2"], B=["Y"])      # C omitted -> unconditional MI
    Sigma_Y = dag.cov("Y")                      # self-covariance block of Y

Correlated roots (Slepian-Wolf / CEO / common-information settings) are declared
with an optional separate method; roots are independent by default::

    dag.add_root_correlation("X1", "X2", cov=Sigma_12)   # cov = E[V_X1 V_X2^H]

Profiles (see builder_implementation.md, spec v0.2): this library implements the
*conditional* (``cmi(A, B, C)``), *multiroot* (more than one source), and a
*correlated-roots* extension. ``add_source`` (rather than "add_root") is kept as
the caller-facing name for cross-library recognizability (spec section 4.3),
matching the "source" / "root" synonymy of the spec; internally a source maps to
a root of the multi-root core.

Matrices may be given either as concrete tensors (used as-is) or by name
(strings resolved at query time via a ``bind={name: tensor}`` mapping). An
unbound name raises ``ValueError`` -- data is never fabricated (spec section 8).

Conventions inherited from the core (``cmi_dag.krecursion``): node indices are
0-based; roots are the prefix ``{0, ..., K-1}`` in topological order; edge keys
are ``(j, i)`` for the edge ``i -> j`` with ``i < j``; cross-root covariances
use canonical keys ``(r, r')`` with ``r > r'``. Canonical node indices are
assigned by a stable topological sort (Kahn's algorithm, FIFO queue, ties broken
by build/call order). Because only sources have in-degree 0, the sources always
receive the contiguous prefix ``0, ..., K-1`` -- exactly the root convention the
core requires.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence, Union

import torch

from cmi_dag.information import conditional_mutual_information_from_k
from cmi_dag.krecursion import compute_k_blocks_multiroot, get_K

# A matrix reference recorded at build time: either a concrete tensor (used
# as-is) or a name (string) resolved at query time via the ``bind`` mapping.
MatrixRef = Union[str, torch.Tensor]


class GaussianDAG:
    """Declarative named-node builder for a multi-root linear Gaussian DAG.

    See the module docstring for the worked example, the supported profiles
    (conditional / multiroot / correlated-roots), and the matrix-binding rules.
    The builder is a thin layer: it records structure and matrix references at
    build time, then lowers to the library's functional core when a query
    (``cmi`` / ``cov``) runs.
    """

    def __init__(self) -> None:
        # name -> root covariance reference (tensor or name string).
        self._sources: dict[str, MatrixRef] = {}
        # name -> (parents: {parent_name: gain_ref}, noise_ref).
        self._nodes: dict[str, tuple[dict[str, MatrixRef], MatrixRef]] = {}
        # (name_a, name_b, cov_ref) with cov = E[V_a V_b^H] for distinct roots.
        self._cross: list[tuple[str, str, MatrixRef]] = []
        # Build/call order; used to derive canonical indices (spec section 12).
        self._order: list[str] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_source(self, name: str, *, cov: MatrixRef) -> "GaussianDAG":
        """Declare a source (root) node with covariance ``cov``.

        This library is multi-root: any number of sources may be declared. By
        default the sources are mutually independent; use
        ``add_root_correlation`` to seed a cross-covariance.

        Returns ``self`` to allow chaining.
        """
        self._check_new(name)
        self._sources[name] = cov
        self._order.append(name)
        return self

    def add_node(
        self,
        name: str,
        *,
        parents: dict[str, MatrixRef],
        noise: MatrixRef,
    ) -> "GaussianDAG":
        """Declare a non-source node from its ``parents`` and own ``noise``.

        ``parents`` maps each parent's name to the gain matrix on that edge.
        Every parent must already be declared (this enforces acyclicity and a
        valid topological order, and catches self-loops). ``parents`` must be
        non-empty -- a parentless node is a source; use ``add_source``.

        Returns ``self`` to allow chaining.
        """
        self._check_new(name)
        if not parents:
            raise ValueError(
                f"Node {name!r} has no parents. A parentless node is a source; "
                "use add_source(name, cov=...)."
            )
        for p in parents:
            if p not in self._sources and p not in self._nodes:
                raise ValueError(
                    f"Unknown parent {p!r} of node {name!r}: declare it with "
                    "add_source/add_node before referencing it."
                )
        self._nodes[name] = (dict(parents), noise)
        self._order.append(name)
        return self

    def add_root_correlation(
        self, name_a: str, name_b: str, *, cov: MatrixRef
    ) -> "GaussianDAG":
        """Declare a cross-covariance ``cov = E[V_a V_b^H]`` between two roots.

        Both names must be declared sources and distinct. At most one
        correlation may be declared per unordered pair. Lowers to the core's
        ``cross_root_covs`` (canonical key ``r > r'``); the Hermitian
        orientation is handled by the builder.

        Returns ``self`` to allow chaining.
        """
        for nm in (name_a, name_b):
            if nm not in self._sources:
                raise ValueError(
                    f"add_root_correlation: {nm!r} is not a declared source."
                )
        if name_a == name_b:
            raise ValueError(
                f"add_root_correlation: a root cannot correlate with itself "
                f"({name_a!r}); its covariance is set via add_source."
            )
        for a, b, _ in self._cross:
            if {a, b} == {name_a, name_b}:
                raise ValueError(
                    f"Duplicate root correlation for pair "
                    f"{{{name_a!r}, {name_b!r}}}."
                )
        self._cross.append((name_a, name_b, cov))
        return self

    def _check_new(self, name: str) -> None:
        if name in self._sources or name in self._nodes:
            raise ValueError(f"Duplicate node {name!r}.")

    # ------------------------------------------------------------------
    # Lowering: names -> canonical indices / core inputs
    # ------------------------------------------------------------------

    def _canonical_index(self) -> dict[str, int]:
        """Assign canonical 0-based indices via a stable topological sort.

        Kahn's algorithm with a FIFO queue; the queue is seeded, and every tie
        broken, by build/call order (``self._order``). Deterministic for a
        given build script (spec section 12). Only sources have in-degree 0, so
        they are enqueued first (in call order) and receive the contiguous
        prefix ``0, ..., K-1`` -- matching the core's "roots are the prefix
        {0, ..., K-1}" requirement.
        """
        children: dict[str, list[str]] = {n: [] for n in self._order}
        indeg: dict[str, int] = {n: 0 for n in self._order}
        for name, (parents, _) in self._nodes.items():
            for p in parents:
                children[p].append(name)
                indeg[name] += 1

        queue: deque[str] = deque(n for n in self._order if indeg[n] == 0)
        index: dict[str, int] = {}
        nxt = 0
        while queue:
            n = queue.popleft()
            index[n] = nxt
            nxt += 1
            for c in children[n]:  # children already in build order
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)

        if len(index) != len(self._order):
            # Unreachable given add_node's pre-declared-parent rule, but guard
            # against a cycle rather than silently dropping nodes.
            raise ValueError("DAG contains a cycle; cannot order nodes.")
        return index

    def _lower_structure(
        self,
    ) -> tuple[list[str], set[int], dict[int, list[int]], set[tuple[int, int]]]:
        """Return the index-based structure (no matrix resolution).

        Yields ``(order, sources, parents, edges)`` -- node names by canonical
        index, the set of source/root indices, the parent-index lists, and the
        ``(child, parent)`` index pairs. Used by the structural-conformance
        checks (spec section 12).
        """
        idx = self._canonical_index()
        order = [name for name, _ in sorted(idx.items(), key=lambda kv: kv[1])]
        sources = {idx[s] for s in self._sources}
        parents = {
            idx[n]: [idx[p] for p in ps] for n, (ps, _) in self._nodes.items()
        }
        edges = {
            (idx[n], idx[p])
            for n, (ps, _) in self._nodes.items()
            for p in ps
        }
        return order, sources, parents, edges

    @staticmethod
    def _resolve(
        m: MatrixRef, bind: dict[str, torch.Tensor] | None
    ) -> torch.Tensor:
        """Resolve a matrix reference to a concrete tensor.

        A concrete tensor is used as-is; a name (string) is looked up in
        ``bind``. An unbound name raises ``ValueError`` -- never fabricated.
        """
        if isinstance(m, str):
            if bind is None or m not in bind:
                raise ValueError(
                    f"Matrix name {m!r} is not bound. Pass it via "
                    "bind={...} on the query."
                )
            return bind[m]
        return m

    def _lower_core_inputs(self, bind: dict[str, torch.Tensor] | None) -> dict:
        """Build the keyword arguments consumed by ``compute_k_blocks_multiroot``,
        resolving every matrix reference."""
        if not self._sources:
            raise ValueError("No source declared; call add_source(...) first.")
        idx = self._canonical_index()

        roots = sorted(idx[s] for s in self._sources)
        parents = {
            idx[n]: [idx[p] for p in ps] for n, (ps, _) in self._nodes.items()
        }
        edge_mats = {
            (idx[n], idx[p]): self._resolve(g, bind)
            for n, (ps, _) in self._nodes.items()
            for p, g in ps.items()
        }
        root_covs = {
            idx[s]: self._resolve(cv, bind) for s, cv in self._sources.items()
        }
        noise_covs = {
            idx[n]: self._resolve(nz, bind) for n, (_, nz) in self._nodes.items()
        }

        cross_root_covs: dict[tuple[int, int], torch.Tensor] = {}
        for a, b, cov in self._cross:
            ra, rb = idx[a], idx[b]
            mat = self._resolve(cov, bind)  # cov = E[V_a V_b^H]
            if ra > rb:
                cross_root_covs[(ra, rb)] = mat
            else:
                # canonical key needs r > r'; E[V_b V_a^H] = (E[V_a V_b^H])^H.
                cross_root_covs[(rb, ra)] = mat.mH

        return dict(
            num_nodes=len(self._order),
            roots=roots,
            parents=parents,
            edge_mats=edge_mats,
            root_covs=root_covs,
            noise_covs=noise_covs,
            cross_root_covs=cross_root_covs or None,
        )

    def _require_known(self, name: str) -> None:
        if name not in self._sources and name not in self._nodes:
            raise ValueError(f"Unknown node {name!r}.")

    # ------------------------------------------------------------------
    # Queries (each lowers to the core and returns its result)
    # ------------------------------------------------------------------

    def cmi(
        self,
        A: Sequence[str],
        B: Sequence[str],
        C: Sequence[str] = (),
        *,
        bind: dict[str, torch.Tensor] | None = None,
        jitter: float = 0.0,
    ) -> torch.Tensor:
        """Conditional mutual information ``I(V_A; V_B | V_C)`` in nats.

        ``A``, ``B``, ``C`` are lists of node names; ``C`` defaults to empty
        (unconditional MI). The non-empty / pairwise-disjoint requirements are
        enforced by the core. Returns a differentiable real scalar tensor,
        exactly what ``conditional_mutual_information_from_k`` returns.
        """
        for nm in (*A, *B, *C):
            self._require_known(nm)
        idx = self._canonical_index()
        K = compute_k_blocks_multiroot(**self._lower_core_inputs(bind))
        return conditional_mutual_information_from_k(
            K,
            A=[idx[n] for n in A],
            B=[idx[n] for n in B],
            C=[idx[n] for n in C],
            jitter=jitter,
        )

    def cov(
        self,
        node: str,
        *,
        bind: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Self-covariance block ``Sigma_node = K_{node,node}``.

        Lowers to ``compute_k_blocks_multiroot`` and returns the canonical
        self-block via ``get_K``. Differentiable through the resolved matrices.
        """
        self._require_known(node)
        idx = self._canonical_index()
        K = compute_k_blocks_multiroot(**self._lower_core_inputs(bind))
        return get_K(K, idx[node], idx[node])
