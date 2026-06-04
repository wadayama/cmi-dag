# gaussian-dag-cmi tutorials

A five-part walkthrough of the library, from a first conditional MI
evaluation on a 2-user MAC to the reproduction of the paper's random
multi-hop multi-source MAC figure.

| # | Topic | File |
| --- | --- | --- |
| 1 | Installation and your first conditional MI | [`tutorial-1-installation-and-first-cmi.md`](tutorial-1-installation-and-first-cmi.md) |
| 2 | Multi-root K-recursion and conditional covariance | [`tutorial-2-multi-root-and-conditional-covariance.md`](tutorial-2-multi-root-and-conditional-covariance.md) |
| 3 | Rate functions and PGA on multi-terminal objectives | [`tutorial-3-rate-functions-and-pga.md`](tutorial-3-rate-functions-and-pga.md) |
| 4 | Sign-indefinite objectives and `pga_descent` (wiretap) | [`tutorial-4-sign-indefinite-and-pga-descent.md`](tutorial-4-sign-indefinite-and-pga-descent.md) |
| 5 | Reproducing the random multi-hop MAC figure | [`tutorial-5-reproducing-random-mac.md`](tutorial-5-reproducing-random-mac.md) |

Each tutorial is self-contained and includes runnable code snippets. The
scripts under `../examples/` accompany Tutorials 3–5 as the polished
end-to-end versions; reading them is optional.

These tutorials assume familiarity with the parent library
[`gaussian-dag`](https://github.com/wadayama/gaussian-dag) — in particular
the single-root `compute_k_blocks` and `pga_ascent` covered in its own
[docs](https://github.com/wadayama/gaussian-dag/tree/main/docs). If you
have not yet, working through that tutorial series first will make this
one substantially easier.
