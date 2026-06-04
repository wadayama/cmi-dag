# `cmi-dag/examples/`

Three self-contained scripts that each (a) reproduce one figure of the
companion paper *"Differentiable Conditional Mutual Information for Linear
Gaussian-DAG Multi-Terminal Networks"* and (b) double as runnable pedagogy
for the library's public API. Each script is independent — no shared
utilities, no configuration files, no separate replot step.

Outputs of compute (`.npz`) and rendered figures (`.pdf`) land in
`results/` and `figures/`, respectively; both directories are git-ignored.

## Quick start

```bash
uv sync --extra examples       # installs matplotlib alongside the core lib
uv run python examples/rate_region_maximization.py
uv run python examples/secure_precoding.py
uv run python examples/random_mac.py
```

Total wall-clock: under 15 seconds on a modern CPU for all three combined.

## Scripts

| Script | Manuscript figure | Demonstrates |
|---|---|---|
| `rate_region_maximization.py` | `rate_region_evolution.pdf` | Joint optimization of two per-user precoders on a 2-user MIMO MAC; the rate-region pentagon expands monotonically over 120 PGA iterations. |
| `secure_precoding.py` | `secure_precoding.pdf` | Sign-indefinite secrecy-rate objective $I(X;Y) - I(X;Z)$ on a MIMO wiretap channel; the precoder shapes its subspace to drive eavesdropper information down while keeping legitimate information high. |
| `random_mac.py` | `random_mac.pdf` | The same MAC-facet-sum objective scaled to a randomly generated 12-node multi-hop multi-source network with 9 relay processing matrices and a single shared total-power budget. |

## Output convention

- **Compute** → `examples/results/<name>.npz` (numpy archive containing the
  full optimization trajectory, the final precoders, the channel
  realization, and a `config` dict).
- **Figure** → `examples/figures/<name>.pdf` (single PDF, regenerated inside
  the same script after the `.npz` save).

Both `results/` and `figures/` are listed in `.gitignore`. Re-running a
script regenerates both files from scratch.

## Reproducibility

Each script hardcodes its seeds and hyperparameters at the top of the file.
Re-running a script with no edits produces a bit-identical `.npz` and PDF.

Channel entries are drawn as standard complex Gaussians ($\mathrm{CN}(0,1)$
i.i.d., real and imaginary parts $\mathcal{N}(0, 1/2)$ each) from a single
PyTorch generator seeded with the script-level `SEED` constant. The
dimensions, power budget, number of iterations, and step size each
script uses match the hyperparameters of the canonical experiment for
the corresponding paper figure; the specific channel realization is
deterministic from the seed in this script and need not coincide with
the realization the paper figure was rendered from. The trajectory
shape and the qualitative behavior of the optimization match.

## Device

All scripts auto-detect CUDA and fall back to CPU; complex128 throughout.
The matrix dimensions are small enough that CPU is more than adequate.
The seeded channel draw happens on CPU so the random sequence is
deterministic across devices.

## Companion library

These examples are the sister set to
[`gaussian-dag/examples/`](https://github.com/wadayama/gaussian-dag/tree/main/examples)
of the parent library. There the topologies are single-root (one
transmitter → MIMO sink); here the topologies are multi-root (multiple
sources) and use the conditional mutual information layer added by
`cmi_dag`.
