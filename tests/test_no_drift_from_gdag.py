"""Drift guard: vendored primitives must stay in sync with gaussian-dag.

`cmi-dag` is fully self-contained and deliberately *vendors* (copies) the
generic numerical primitives it shares with the parent library
`gaussian-dag`. The upside is zero runtime dependency; the cost is that a
fix to a shared primitive must be applied in *both* repositories. This test
is the guardrail against forgetting: it checks that each vendored function's
**numerical logic** is identical to the original in `gaussian-dag`.

Sync surface checked (8 functions):

    cmi_dag/krecursion.py   : hermitianize, get_K
    cmi_dag/information.py  : logdet_hpd
    cmi_dag/optimize.py     : pga_ascent
    cmi_dag/projections.py  : project_frobenius_ball, project_total_power
    tests/gdag_reference.py : compute_k_blocks, mutual_information_from_k
                              (the single-root cross-validation oracle)

How the comparison works:

- We read both source trees *from disk* and compare with `ast`. We never
  import `gaussian_dag`, so this test does not reintroduce a dependency and
  the standalone property is preserved.
- We compare the function body with the docstring removed and every *string
  literal* replaced by a placeholder. This means diagnostic / error-message
  text may differ between the two copies (cmi-dag intentionally localizes a
  few messages to its own API names, e.g. `pga_descent`,
  `conditional_mutual_information_from_k`), while any change to the actual
  computation — operators, numeric constants, control flow — is caught.

Developer environment only: the test needs the `gaussian-dag` source checked
out next to this repo (or pointed to by the `GDAG_SRC` environment variable).
When it is not found (e.g. an end user who cloned only `cmi-dag`), the test
*skips* — it never fails for lack of the parent.

When this test fails it means gaussian-dag and cmi-dag have diverged in the
numerical logic of a shared primitive. Re-sync the two copies (port the
change in whichever direction is correct) and re-run.
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _find_gdag_pkg() -> pathlib.Path | None:
    """Locate the `gaussian_dag` source package, or return None if absent."""
    candidates = []
    env = os.environ.get("GDAG_SRC")
    if env:
        p = pathlib.Path(env)
        # Allow GDAG_SRC to point either at the package dir or the repo root.
        candidates += [p, p / "gaussian_dag"]
    # Sibling checkouts: local dir name (`gaussian-dag-public`) or the
    # GitHub repo name (`gaussian-dag`).
    for name in ("gaussian-dag-public", "gaussian-dag"):
        candidates.append(REPO_ROOT.parent / name / "gaussian_dag")
    for c in candidates:
        if (c / "krecursion.py").is_file():
            return c
    return None


GDAG_PKG = _find_gdag_pkg()

pytestmark = pytest.mark.skipif(
    GDAG_PKG is None,
    reason="gaussian-dag source not found next to this repo (set GDAG_SRC to "
    "enable the drift guard); skipping — standalone clones do not need it.",
)

# (function name, gaussian_dag source file, cmi-dag source file relative to repo)
SPECS = [
    ("hermitianize", "krecursion.py", "cmi_dag/krecursion.py"),
    ("get_K", "krecursion.py", "cmi_dag/krecursion.py"),
    ("logdet_hpd", "information.py", "cmi_dag/information.py"),
    ("pga_ascent", "optimize.py", "cmi_dag/optimize.py"),
    ("project_frobenius_ball", "projections.py", "cmi_dag/projections.py"),
    ("project_total_power", "projections.py", "cmi_dag/projections.py"),
    ("compute_k_blocks", "krecursion.py", "tests/gdag_reference.py"),
    ("mutual_information_from_k", "information.py", "tests/gdag_reference.py"),
    ("compute_effective_channel", "krecursion.py", "tests/gdag_reference.py"),
]


class _BlankStrings(ast.NodeTransformer):
    """Replace every string-constant with a placeholder.

    Numeric constants (ints, floats, complex) are left intact, so a change to
    a computed value is still detected; only human-readable text is ignored.
    """

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="<STR>"), node)
        return node


def _normalized_logic(source_file: pathlib.Path, func_name: str) -> str | None:
    """Return a string-stripped, docstring-free AST dump of `func_name`.

    Returns None if the function is not defined in the file.
    """
    src = source_file.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # Drop a leading docstring statement, if present.
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
            node = _BlankStrings().visit(node)
            ast.fix_missing_locations(node)
            # include_attributes=False -> line numbers ignored, structure only.
            return ast.dump(node)
    return None


@pytest.mark.parametrize(
    ("func_name", "gdag_file", "cmi_file"),
    SPECS,
    ids=[s[0] for s in SPECS],
)
def test_vendored_primitive_matches_gdag(
    func_name: str, gdag_file: str, cmi_file: str
) -> None:
    assert GDAG_PKG is not None  # guarded by pytestmark
    gdag_src = GDAG_PKG / gdag_file
    cmi_src = REPO_ROOT / cmi_file

    gdag_logic = _normalized_logic(gdag_src, func_name)
    cmi_logic = _normalized_logic(cmi_src, func_name)

    assert gdag_logic is not None, (
        f"`{func_name}` not found in gaussian-dag at {gdag_src}. "
        "The parent may have moved or renamed it; update SPECS / re-sync."
    )
    assert cmi_logic is not None, (
        f"`{func_name}` not found in cmi-dag at {cmi_src}."
    )
    assert gdag_logic == cmi_logic, (
        f"Numerical-logic DRIFT in vendored `{func_name}`: "
        f"{cmi_file} no longer matches gaussian-dag/{gdag_file} "
        "(string literals are ignored, so this is a real computation change). "
        "Re-sync the two copies."
    )
