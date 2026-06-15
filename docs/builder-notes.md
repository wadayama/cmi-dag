# Builder notes — cmi-dag

Library-specific decisions for the named-node DAG builder, per the template in
`builder_implementation.md` §13. The shared policy lives in that document; only
the choices it delegates to the implementer are recorded here.

- **Conforms to builder_implementation.md spec version:** 0.2
- **Profiles implemented:** `conditional` (CMI with a conditioning set),
  `multiroot` (more than one source), and a `correlated-roots` extension
  (`cross_root_covs`). The `stochastic/batch` profile is not implemented.
- **Query method(s):**
  - `cmi(A, B, C=(), *, bind=None, jitter=0.0)` — `I(V_A; V_B | V_C)` in nats,
    where `A`, `B`, `C` are lists of node names (`C` optional).
  - `cov(node, *, bind=None)` — the self-covariance block `Σ_node = K_{node,node}`.
  - Returns whatever the core returns: a differentiable real-scalar tensor
    (`cmi`) or a covariance-block tensor (`cov`).
- **Class name:** `GaussianDAG` — matches the worked example in spec §3 and the
  multiroot illustration in §7, and is the same class name used by the sibling
  `gaussian-dag` builder, for maximal cross-library recognizability. No
  collision: the name did not exist in the repo, and it is added to
  `cmi_dag/__init__.__all__` alongside (never replacing) the existing 13 public
  symbols.
- **Matrix input:** both **by name (string)** and **as a concrete tensor**. A
  tensor is used as-is; a name is resolved at query time via a
  `bind={name: tensor}` mapping passed to the query. An unbound name raises
  `ValueError` — data is never fabricated (spec §8). No batch axis.
- **Matrix conventions:** inherited from the core. `complex128` is the standard
  dtype; tensors keep their own dtype/device (device-agnostic). Edge keys lower
  to `(j, i)` for the edge `i → j` with `i < j`; roots are the prefix
  `{0, …, K-1}`. Cross-root covariances use canonical keys `(r, r')` with
  `r > r'`; the builder orients the user-supplied `cov = E[V_a V_b^H]`
  accordingly (Hermitian-transposing when the canonical key flips the pair).
- **Module / namespace:** `cmi_dag/builder.py`, re-exported from
  `cmi_dag/__init__.py`.
- **Canonical index (spec §12):** stable topological sort — Kahn's algorithm
  with a FIFO queue, seeding and tie-breaking by build/call order. Only sources
  have in-degree 0, so they receive the contiguous prefix `0, …, K-1`, which is
  exactly the multi-root core's root convention
  (`compute_k_blocks_multiroot` requires `roots == {0, …, K-1}`). Exposed for
  structural tests via the internal `_lower_structure()`.
- **Deliberate divergences from the recommended idioms (§5) and why:**
  - **`add_source` keeps the spec name even though the core says "root".** The
    spec (§3, §4.3) treats "source" and "root" as synonyms and fixes
    `add_source` as the caller-facing name for cross-library recognizability; a
    source maps internally to a root.
  - **Correlated roots via a separate method, not the core call shape.** Per §7,
    capability is added *around* the recognizable `add_source`/`add_node`/query
    shape: `add_root_correlation(a, b, cov=…)` rather than an extra positional
    argument.
- **Unsupported / invalid constructs and the errors raised:**
  - An `add_node` referencing an undeclared parent (also catches self-loops) →
    `ValueError` ("Unknown parent").
  - A parentless `add_node` → `ValueError` (a source uses `add_source`).
  - A duplicate node name → `ValueError` ("Duplicate").
  - `add_root_correlation` with an undeclared source, a self-pair, or a
    duplicate pair → `ValueError`.
  - An unbound matrix name at query time → `ValueError` ("not bound").
  - `cmi` with non-disjoint or empty `A`/`B` → `ValueError` (enforced by the
    core, `conditional_mutual_information_from_k`).
  - A query on a sources-only DAG (no non-root node) → `ValueError`
    ("num_roots …"), surfaced by the core's `num_roots < num_nodes` check.

## Structural-conformance vectors (spec §12)

`chain`, the `two-source (MAC-like)` graph, and `diamond` are all expressible
and verified by `tests/test_builder.py::test_structure_chain` /
`test_structure_two_source_mac` / `test_structure_diamond`. Unlike a single-root
library, cmi-dag expresses the two-source vector directly (it is the headline
case).
