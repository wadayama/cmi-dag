"""Unit tests for cmi_dag.builder (named-node DAG builder).

Covers the acceptance criteria of builder_implementation.md (spec v0.2):
a chain, multi-root (MAC) and multi-parent graphs, correlated roots, the
round-trip equivalence to the functional core, the structural-conformance
vectors (section 12), name/object binding, and the loud-failure requirements.
"""

from __future__ import annotations

import pytest
import torch

from cmi_dag import GaussianDAG
from cmi_dag.information import conditional_mutual_information_from_k
from cmi_dag.krecursion import compute_k_blocks_multiroot, get_K


DTYPE = torch.complex128


# ----------------------------- helpers -----------------------------


def _randn_complex(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(*shape, dtype=torch.float64, generator=g)
    imag = torch.randn(*shape, dtype=torch.float64, generator=g)
    return torch.complex(real, imag)


def _hermitian_psd(d: int, *, seed: int) -> torch.Tensor:
    A = _randn_complex(d, d, seed=seed)
    return A @ A.mH + torch.eye(d, dtype=DTYPE)


# ============================================================
# 2-user MAC round-trip (the headline multiroot + conditional case)
# ============================================================


def test_mac_cmi_roundtrip():
    d = 2
    S1 = _hermitian_psd(d, seed=100)
    S2 = _hermitian_psd(d, seed=101)
    H1 = _randn_complex(d, d, seed=1)
    H2 = _randn_complex(d, d, seed=2)
    N_Y = _hermitian_psd(d, seed=200)

    dag = GaussianDAG()
    dag.add_source("X1", cov=S1)
    dag.add_source("X2", cov=S2)
    dag.add_node("Y", parents={"X1": H1, "X2": H2}, noise=N_Y)

    # Hand-built core (X1=0, X2=1, Y=2).
    K = compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0, 1],
        parents={2: [0, 1]},
        edge_mats={(2, 0): H1, (2, 1): H2},
        root_covs={0: S1, 1: S2},
        noise_covs={2: N_Y},
    )

    # Three MAC facets.
    assert torch.allclose(
        dag.cmi(A=["X1"], B=["Y"], C=["X2"]),
        conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1]),
        atol=1e-10,
    )
    assert torch.allclose(
        dag.cmi(A=["X2"], B=["Y"], C=["X1"]),
        conditional_mutual_information_from_k(K, A=[1], B=[2], C=[0]),
        atol=1e-10,
    )
    assert torch.allclose(
        dag.cmi(A=["X1", "X2"], B=["Y"]),  # C omitted -> unconditional
        conditional_mutual_information_from_k(K, A=[0, 1], B=[2], C=[]),
        atol=1e-10,
    )


# ============================================================
# chain X -> Y -> Z : round-trip (single root, conditional)
# ============================================================


def test_chain_cmi_roundtrip():
    d = 2
    Sx = _hermitian_psd(d, seed=100)
    H_XY = _randn_complex(d, d, seed=1)
    H_YZ = _randn_complex(d, d, seed=2)
    N_Y = _hermitian_psd(d, seed=201)
    N_Z = _hermitian_psd(d, seed=202)

    dag = GaussianDAG()
    dag.add_source("X", cov=Sx)
    dag.add_node("Y", parents={"X": H_XY}, noise=N_Y)
    dag.add_node("Z", parents={"Y": H_YZ}, noise=N_Z)

    K = compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0],
        parents={1: [0], 2: [1]},
        edge_mats={(1, 0): H_XY, (2, 1): H_YZ},
        root_covs={0: Sx},
        noise_covs={1: N_Y, 2: N_Z},
    )
    assert torch.allclose(
        dag.cmi(A=["X"], B=["Z"], C=["Y"]),
        conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1]),
        atol=1e-10,
    )
    assert torch.allclose(
        dag.cmi(A=["X"], B=["Z"]),
        conditional_mutual_information_from_k(K, A=[0], B=[2], C=[]),
        atol=1e-10,
    )
    assert torch.allclose(dag.cov("Z"), get_K(K, 2, 2), atol=1e-10)


# ============================================================
# diamond X -> {Y, W} -> Z : round-trip (single root, multi-parent)
# ============================================================


def test_diamond_cmi_roundtrip():
    d = 2
    Sx = _hermitian_psd(d, seed=100)
    A_XY = _randn_complex(d, d, seed=1)
    A_XW = _randn_complex(d, d, seed=2)
    A_YZ = _randn_complex(d, d, seed=3)
    A_WZ = _randn_complex(d, d, seed=4)
    N_Y = _hermitian_psd(d, seed=201)
    N_W = _hermitian_psd(d, seed=202)
    N_Z = _hermitian_psd(d, seed=203)

    dag = GaussianDAG()
    dag.add_source("X", cov=Sx)
    dag.add_node("Y", parents={"X": A_XY}, noise=N_Y)
    dag.add_node("W", parents={"X": A_XW}, noise=N_W)
    dag.add_node("Z", parents={"Y": A_YZ, "W": A_WZ}, noise=N_Z)

    K = compute_k_blocks_multiroot(
        num_nodes=4,
        roots=[0],
        parents={1: [0], 2: [0], 3: [1, 2]},
        edge_mats={(1, 0): A_XY, (2, 0): A_XW, (3, 1): A_YZ, (3, 2): A_WZ},
        root_covs={0: Sx},
        noise_covs={1: N_Y, 2: N_W, 3: N_Z},
    )
    assert torch.allclose(
        dag.cmi(A=["X"], B=["Z"]),
        conditional_mutual_information_from_k(K, A=[0], B=[3], C=[]),
        atol=1e-10,
    )


# ============================================================
# Correlated roots round-trip (cross_root_covs, canonical r > r')
# ============================================================


def test_correlated_roots_roundtrip():
    d = 2
    S1 = _hermitian_psd(d, seed=100)
    S2 = _hermitian_psd(d, seed=101)
    # A small cross-covariance E[V_X1 V_X2^H] keeping the joint PD.
    S12 = 0.1 * _randn_complex(d, d, seed=5)
    H1 = _randn_complex(d, d, seed=1)
    H2 = _randn_complex(d, d, seed=2)
    N_Y = _hermitian_psd(d, seed=200)

    dag = GaussianDAG()
    dag.add_source("X1", cov=S1)
    dag.add_source("X2", cov=S2)
    dag.add_root_correlation("X1", "X2", cov=S12)  # cov = E[V_X1 V_X2^H]
    dag.add_node("Y", parents={"X1": H1, "X2": H2}, noise=N_Y)

    # Canonical: X1=0, X2=1, so the (r>r') key is (1, 0) = E[V_X2 V_X1^H] = S12^H.
    K = compute_k_blocks_multiroot(
        num_nodes=3,
        roots=[0, 1],
        parents={2: [0, 1]},
        edge_mats={(2, 0): H1, (2, 1): H2},
        root_covs={0: S1, 1: S2},
        noise_covs={2: N_Y},
        cross_root_covs={(1, 0): S12.mH},
    )
    assert torch.allclose(
        dag.cmi(A=["X1"], B=["Y"], C=["X2"]),
        conditional_mutual_information_from_k(K, A=[0], B=[2], C=[1]),
        atol=1e-10,
    )
    # The off-diagonal root block itself matches (orientation check).
    assert torch.allclose(dag.cov("X2"), S2, atol=1e-10)


def test_root_correlation_orientation_independent_of_arg_order():
    # Declaring (X2, X1) with cov = E[V_X2 V_X1^H] must yield the same DAG as
    # declaring (X1, X2) with the Hermitian-transposed cov.
    d = 2
    S1 = _hermitian_psd(d, seed=100)
    S2 = _hermitian_psd(d, seed=101)
    S12 = 0.1 * _randn_complex(d, d, seed=5)  # E[V_X1 V_X2^H]
    H1 = _randn_complex(d, d, seed=1)
    H2 = _randn_complex(d, d, seed=2)
    N_Y = _hermitian_psd(d, seed=200)

    def _build(corr_args):
        dag = GaussianDAG()
        dag.add_source("X1", cov=S1)
        dag.add_source("X2", cov=S2)
        dag.add_root_correlation(*corr_args[0], cov=corr_args[1])
        dag.add_node("Y", parents={"X1": H1, "X2": H2}, noise=N_Y)
        return dag.cmi(A=["X1"], B=["Y"], C=["X2"])

    mi_ab = _build((("X1", "X2"), S12))
    mi_ba = _build((("X2", "X1"), S12.mH))  # E[V_X2 V_X1^H] = S12^H
    assert torch.allclose(mi_ab, mi_ba, atol=1e-12)


# ============================================================
# Structural-conformance vectors (spec section 12, structure only)
# ============================================================


def test_structure_chain():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    dag.add_node("Y", parents={"X": "H_XY"}, noise="N_Y")
    dag.add_node("Z", parents={"Y": "H_YZ"}, noise="N_Z")
    order, sources, parents, edges = dag._lower_structure()
    assert order == ["X", "Y", "Z"]
    assert sources == {0}
    assert parents == {1: [0], 2: [1]}
    assert edges == {(1, 0), (2, 1)}


def test_structure_two_source_mac():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    dag.add_source("Y", cov="Sy")
    dag.add_node("Z", parents={"X": "H_XZ", "Y": "H_YZ"}, noise="N_Z")
    order, sources, parents, edges = dag._lower_structure()
    assert order == ["X", "Y", "Z"]
    assert sources == {0, 1}
    assert parents == {2: [0, 1]}
    assert edges == {(2, 0), (2, 1)}


def test_structure_diamond():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    dag.add_node("Y", parents={"X": "H_XY"}, noise="N_Y")
    dag.add_node("W", parents={"X": "H_XW"}, noise="N_W")
    dag.add_node("Z", parents={"Y": "H_YZ", "W": "H_WZ"}, noise="N_Z")
    order, sources, parents, edges = dag._lower_structure()
    assert order == ["X", "Y", "W", "Z"]
    assert sources == {0}
    assert parents == {1: [0], 2: [0], 3: [1, 2]}
    assert edges == {(1, 0), (2, 0), (3, 1), (3, 2)}


# ============================================================
# Binding: name-or-object resolved at query time (spec section 8)
# ============================================================


def test_bind_by_name_matches_concrete():
    d = 2
    S1 = _hermitian_psd(d, seed=100)
    S2 = _hermitian_psd(d, seed=101)
    H1 = _randn_complex(d, d, seed=1)
    H2 = _randn_complex(d, d, seed=2)
    N_Y = _hermitian_psd(d, seed=200)
    binding = {"S1": S1, "S2": S2, "H1": H1, "H2": H2, "N_Y": N_Y}

    by_name = GaussianDAG()
    by_name.add_source("X1", cov="S1")
    by_name.add_source("X2", cov="S2")
    by_name.add_node("Y", parents={"X1": "H1", "X2": "H2"}, noise="N_Y")
    mi_named = by_name.cmi(A=["X1"], B=["Y"], C=["X2"], bind=binding)

    by_obj = GaussianDAG()
    by_obj.add_source("X1", cov=S1)
    by_obj.add_source("X2", cov=S2)
    by_obj.add_node("Y", parents={"X1": H1, "X2": H2}, noise=N_Y)
    mi_obj = by_obj.cmi(A=["X1"], B=["Y"], C=["X2"])

    assert torch.allclose(mi_named, mi_obj, atol=1e-12)


def test_unbound_name_raises():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    dag.add_node("Y", parents={"X": "H_XY"}, noise="N_Y")
    with pytest.raises(ValueError, match="not bound"):
        dag.cmi(A=["X"], B=["Y"], bind={"Sx": _hermitian_psd(2, seed=1)})


# ============================================================
# Differentiability survives the builder (core autograd preserved)
# ============================================================


def test_cmi_is_differentiable_through_builder():
    d = 2
    S1 = torch.eye(d, dtype=DTYPE)
    S2 = torch.eye(d, dtype=DTYPE)
    H1 = _randn_complex(d, d, seed=1)
    F = (0.1 * _randn_complex(d, d, seed=7)).requires_grad_(True)
    H2 = _randn_complex(d, d, seed=2)
    N_Y = _hermitian_psd(d, seed=200)

    dag = GaussianDAG()
    dag.add_source("X1", cov=S1)
    dag.add_source("X2", cov=S2)
    dag.add_node("Y", parents={"X1": H1 @ F, "X2": H2}, noise=N_Y)
    mi = dag.cmi(A=["X1"], B=["Y"], C=["X2"])
    mi.backward()

    assert F.grad is not None
    assert torch.isfinite(F.grad).all()


# ============================================================
# Loud failures for unsupported / invalid constructs (spec section 4.4)
# ============================================================


def test_unknown_parent_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    with pytest.raises(ValueError, match="Unknown parent"):
        dag.add_node("Z", parents={"Q": "H"}, noise="N_Z")


def test_duplicate_name_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    with pytest.raises(ValueError, match="Duplicate"):
        dag.add_source("X", cov="Sx2")


def test_parentless_node_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    with pytest.raises(ValueError, match="no parents"):
        dag.add_node("Y", parents={}, noise="N_Y")


def test_self_loop_rejected():
    dag = GaussianDAG()
    dag.add_source("X", cov="Sx")
    with pytest.raises(ValueError, match="Unknown parent"):
        dag.add_node("Y", parents={"Y": "H"}, noise="N_Y")


def test_root_correlation_unknown_source_rejected():
    dag = GaussianDAG()
    dag.add_source("X1", cov="S1")
    with pytest.raises(ValueError, match="not a declared source"):
        dag.add_root_correlation("X1", "X2", cov="S12")


def test_root_correlation_self_rejected():
    dag = GaussianDAG()
    dag.add_source("X1", cov="S1")
    with pytest.raises(ValueError, match="cannot correlate with itself"):
        dag.add_root_correlation("X1", "X1", cov="S11")


def test_root_correlation_duplicate_pair_rejected():
    dag = GaussianDAG()
    dag.add_source("X1", cov="S1")
    dag.add_source("X2", cov="S2")
    dag.add_root_correlation("X1", "X2", cov="S12")
    with pytest.raises(ValueError, match="Duplicate root correlation"):
        dag.add_root_correlation("X2", "X1", cov="S21")


def test_cmi_overlapping_sets_rejected():
    # A and B sharing a node must fail (disjointness enforced by the core).
    d = 2
    dag = GaussianDAG()
    dag.add_source("X1", cov=_hermitian_psd(d, seed=100))
    dag.add_source("X2", cov=_hermitian_psd(d, seed=101))
    dag.add_node(
        "Y",
        parents={"X1": _randn_complex(d, d, seed=1),
                 "X2": _randn_complex(d, d, seed=2)},
        noise=_hermitian_psd(d, seed=200),
    )
    with pytest.raises(ValueError, match="disjoint"):
        dag.cmi(A=["X1"], B=["X1"], C=["X2"])


def test_sources_only_query_rejected():
    # No non-root node -> the channel is trivial; the core rejects it.
    dag = GaussianDAG()
    dag.add_source("X1", cov=_hermitian_psd(2, seed=100))
    dag.add_source("X2", cov=_hermitian_psd(2, seed=101))
    with pytest.raises(ValueError, match="num_roots"):
        dag.cmi(A=["X1"], B=["X2"])
