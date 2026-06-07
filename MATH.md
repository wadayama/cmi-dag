# Mathematical Foundations

> Mathematical model that this library implements: a closed-form,
> end-to-end differentiable evaluation of *conditional* mutual
> information $I(V_A; V_B \mid V_C)$ for arbitrary disjoint node
> subsets $A, B, C$ of a multi-root linear Gaussian DAG. The K-recursion
> of [`gaussian-dag`](https://github.com/wadayama/gaussian-dag) is
> extended to multiple independent input roots; every node-pair
> covariance is produced in a single forward sweep; and any conditional
> mutual information is then a $\log\det$ difference of sub-block
> Schur complements of the support covariance. Every differentiable
> function of such conditional mutual informations — weighted sum-rate,
> rate-region rate functions of the multiple-access pentagon /
> broadcast / interference / relay regions, and composite sigmoid
> surrogates of rate-region outage — inherits the same computation
> graph and is optimized by the projected gradient method (PGM) with
> one reverse-mode autograd sweep.
>
> This document summarizes the theoretical sections (§II–§III) of the
> accompanying manuscript. The implementation conventions, public API,
> examples, and tutorials are described in [`README.md`](README.md) and
> the tutorials under [`docs/`](docs/).

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Multi-root linear Gaussian DAG and the K-recursion](#2-multi-root-linear-gaussian-dag-and-the-k-recursion)
3. [Block extraction and closed-form conditional MI](#3-block-extraction-and-closed-form-conditional-mi)
4. [Conditional-MI objectives](#4-conditional-mi-objectives)
5. [Multi-terminal rate-region functions](#5-multi-terminal-rate-region-functions)
6. [Composite sigmoid surrogate of rate-region outage](#6-composite-sigmoid-surrogate-of-rate-region-outage)
7. [Optimization by the projected gradient method](#7-optimization-by-the-projected-gradient-method)

---

## 1. Purpose and scope

The companion library
[`gaussian-dag`](https://github.com/wadayama/gaussian-dag) solves
*single-pair* mutual-information optimization on a linear Gaussian DAG
with one input root: a single $I(V_0; V_M)$ is differentiated
end-to-end through a K-recursion forward pass.

`cmi-dag` extends that framework in two directions essential for
multi-terminal network design:

- **Multiple independent input roots.** The DAG carries $K \geq 1$
  independent user-input roots $V_1, \ldots, V_K$ (one per
  rate-bearing user), each with its own input covariance $\Sigma_r$.
  The K-recursion is generalized so that the source-pair base case is
  block-diagonal across roots.
- **Conditional mutual information for arbitrary disjoint subsets.**
  For any disjoint $A, B, C \subseteq \mathcal{V}$ the conditional
  MI $I(V_A; V_B \mid V_C)$ is a $\log\det$ difference of sub-block
  Schur complements of the support covariance — a procedure built
  entirely from differentiable primitives.

From these two ingredients the library evaluates, by chain-rule
composition, every standard multi-terminal design objective: weighted
sum-rate and fairness composites, the rate functions delimiting the
multiple-access pentagon, the Gaussian broadcast region with
dirty-paper coding, the Han–Kobayashi inner bound for interference
channels, decode-/compress-and-forward relay bounds, and a composite
sigmoid surrogate of rate-region outage. All of them are optimized by
projected gradient ascent / descent with a single reverse-mode
autograd sweep — no per-objective gradient derivation is required.

The point-to-point single-root special case recovers
`gaussian-dag` exactly.

---

## 2. Multi-root linear Gaussian DAG and the K-recursion

### 2.1 Notation

$A^{\mathsf{H}}$, $A^*$, $\mathrm{tr}(\cdot)$, $\det(\cdot)$,
$\| \cdot \|_F$, and $I_d$ denote the Hermitian transpose,
entry-wise complex conjugate, trace, determinant, Frobenius norm,
and $d \times d$ identity. $\Sigma \succ 0$
(resp. $\succeq 0$) means Hermitian positive (semi)definite.
$\mathcal{CN}(\mu, \Sigma)$ denotes the circular complex Gaussian
distribution. $\sigma_\tau(x) := (1 + e^{-x/\tau})^{-1}$ is the
sigmoid with temperature $\tau > 0$. All MI values are in **nats**.

### 2.2 DAG, roots, and structural equations

Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a topologically
ordered DAG with $\mathcal{V} = \{1, \ldots, M\}$. The first $K$
nodes are the **user-input roots**,

$$\mathcal{R} = \{1, \ldots, K\} \subsetneq \mathcal{V},$$

one per rate-bearing user; node $r \in \mathcal{R}$ carries the input
signal of the rate-$R_r$ user. Each node holds a complex random
vector $V_j \in \mathbb{C}^{d_j}$. For a root $r \in \mathcal{R}$,

$$V_r \sim \mathcal{CN}(0, \Sigma_r), \qquad r \in \mathcal{R}, \tag{2.1}$$

and the user inputs $\{V_r\}_{r \in \mathcal{R}}$ are **mutually
independent**. For each non-root node $j$ with parents $\mathrm{Pa}(j)$,

$$V_j = \sum_{i \in \mathrm{Pa}(j)} A_{ji} V_i + Z_j, \qquad Z_j \sim \mathcal{CN}(0, \Sigma_j), \tag{2.2}$$

with edge matrices $A_{ji} \in \mathbb{C}^{d_j \times d_i}$ and
independent additive noises $\{Z_j\}$.

### 2.3 Edge factorization and design parameter

Each edge matrix admits a multiplicative factorization

$$A_{ji} = A_{ji}^{(1)} A_{ji}^{(2)} \cdots A_{ji}^{(L_{ji})}, \tag{2.3}$$

each factor labeled either *controllable* (a precoder, relay matrix,
beamformer, ...) or *constant* (a channel realization). The
**design parameter** is the tuple of controllable factors,

$$\eta := \{ A_{ji}^{(\ell)} : (j, i, \ell) \in \mathcal{C} \},$$

optimized over a **feasible set** $\mathcal{F}$ encoding power and
structural constraints (Frobenius per-user balls, shared total power,
unit-modulus RIS, per-antenna power, etc.).

### 2.4 The multi-root K-recursion

The K-recursion constructs every node-pair covariance

$$K_{jk} := \mathbb{E}[V_j V_k^{\mathsf{H}}], \qquad j, k \in \mathcal{V},$$

in topological order from the multi-root base case

$$K_{rr} = \Sigma_r \;\; (r \in \mathcal{R}), \qquad K_{r r'} = 0 \;\; (r \neq r', \; r, r' \in \mathcal{R}). \tag{2.4}$$

For each non-root node $j$ and $k \leq j$,

$$K_{jk} = \begin{cases} \sum_{i \in \mathrm{Pa}(j)} A_{ji} K_{ik} & k < j, \\ \sum_{i, i' \in \mathrm{Pa}(j)} A_{ji} K_{i i'} A_{j i'}^{\mathsf{H}} + \Sigma_j & k = j, \end{cases} \tag{2.5}$$

with the Hermitian-flip convention $K_{ab} = K_{ba}^{\mathsf{H}}$ for
$a < b$. Each step uses only matrix products, sums, and Hermitian
transposes, so the full set $\{K_{jk}\}_{j \geq k}$ is a smooth
function of $\eta$ delivered in a single forward sweep —
`compute_k_blocks_multiroot` in `cmi_dag.krecursion`.

The single-root case $K = 1$ reduces to the original
`gaussian-dag` K-recursion.

---

## 3. Block extraction and closed-form conditional MI

We now evaluate any conditional mutual information on $\mathcal{G}$
directly from the K-blocks.

### 3.1 Support, block extraction, and Schur complements

Let $A, B, C$ be **disjoint** subsets of $\mathcal{V}$ whose
conditional MI we want to evaluate. The **support** of the
computation is

$$S := A \cup B \cup C = \{s_1, \ldots, s_n\}. \tag{3.1}$$

For any $X \subseteq \mathcal{V}$ define the stacked vector
$V_X := [V_{x_1}^{\top}, \ldots, V_{x_{|X|}}^{\top}]^{\top}$.
Two subscript conventions are used throughout:

- juxtaposition of set symbols is set union: $XY := X \cup Y$;
- a comma in a covariance subscript separates row and column index sets,
  $\Sigma_{X, Y} := \mathbb{E}[V_X V_Y^{\mathsf{H}}] = [K_{x_i y_j}]_{x_i \in X, \, y_j \in Y}$.

The **support block covariance** is the $n \times n$ block matrix

$$\Sigma_{S, S} = [K_{s_i s_j}]_{1 \leq i, j \leq n}, \tag{3.2}$$

formed by stacking only the $n^2$ K-blocks indexed by $S$. We call
this assembly **block extraction**. Every row/column sub-block
($\Sigma_{A, A}, \Sigma_{A, C}, \Sigma_{C, C}, \Sigma_{A, BC}, \Sigma_{BC, BC}, \ldots$)
needed below is read off from (3.2) by selecting index ranges.

Recall that for a Hermitian positive-definite block matrix
$M = \begin{bmatrix} P & R \\ R^{\mathsf{H}} & Q \end{bmatrix}$ with
$Q \succ 0$, the **Schur complement** of $Q$ in $M$ is
$P - R Q^{-1} R^{\mathsf{H}}$. When $M$ is the covariance of a
jointly Gaussian pair, this Schur complement is exactly the
conditional covariance.

### 3.2 The closed-form CMI proposition

**Proposition (Closed-form conditional MI on the K-recursion).**
*Let $\mathcal{G}$ be a multi-root linear Gaussian DAG with K-blocks
produced by (2.5), and let $A, B, C \subseteq \mathcal{V}$ be
disjoint with support $S = A \cup B \cup C$ and support block
covariance $\Sigma_{S, S}$ of (3.2). Assume $\Sigma_{C, C} \succ 0$
and $\Sigma_{BC, BC} \succ 0$. Then:*

*(a) The conditional covariances of $V_A$ given $V_C$ and given
$V_{BC}$ are sub-block Schur complements of $\Sigma_{S, S}$,*

$$\Sigma_{A \mid C} = \Sigma_{A, A} - \Sigma_{A, C} \Sigma_{C, C}^{-1} \Sigma_{C, A}, \tag{3.3a}$$

$$\Sigma_{A \mid BC} = \Sigma_{A, A} - \Sigma_{A, BC} \Sigma_{BC, BC}^{-1} \Sigma_{BC, A}. \tag{3.3b}$$

*(b) The conditional mutual information of $(V_A, V_B)$ given $V_C$
admits the log-determinant closed form*

$$I(V_A; V_B \mid V_C) = \log\det \Sigma_{A \mid C} - \log\det \Sigma_{A \mid BC}. \tag{3.4}$$

**Proof sketch.** *(a)* For any disjoint $A, X$ with $\Sigma_{X, X} \succ 0$,
$(V_A, V_X)$ is jointly circular complex Gaussian with covariance
$\Sigma_{AX, AX}$, a sub-block of $\Sigma_{S, S}$; the Schur-complement /
conditional-covariance correspondence yields
$\Sigma_{A \mid X} = \Sigma_{A, A} - \Sigma_{A, X} \Sigma_{X, X}^{-1} \Sigma_{X, A}$.
Specializing to $X = C$ and $X = BC$ gives (3.3).

*(b)* By the entropy chain rule,
$I(V_A; V_B \mid V_C) = h(V_A \mid V_C) - h(V_A \mid V_{BC})$.
A circular complex Gaussian with PD covariance $\Sigma$ has differential
entropy $h = \log\det(\pi e \, \Sigma)$ in nats; substituting (3.3a) and
(3.3b), the $\log\det(\pi e \, I)$ terms cancel and (3.4) follows.
$\square$

### 3.3 Numerical implementation

The inverses $\Sigma_{C, C}^{-1}$ and $\Sigma_{BC, BC}^{-1}$ in (3.3) are
never formed explicitly. Each sub-block Schur complement is evaluated
through a Cholesky-based linear solve, and each log-determinant in
(3.4) as the sum of the diagonal logs of the corresponding Cholesky
factor. Near singularity a small diagonal jitter $\varepsilon I$ is
added before factorization; accumulated floating-point error can drift
$\Sigma_{A \mid C}$ and $\Sigma_{A \mid BC}$ off the Hermitian PD cone,
so each is symmetrized as $\tfrac{1}{2}(\Sigma + \Sigma^{\mathsf{H}})$
before its log-determinant is taken. The relevant code path is
`conditional_mutual_information_from_k` in `cmi_dag.information`.

### 3.4 Differentiability via complex autograd

The K-recursion (2.5), block extraction (3.2), Schur complements (3.3),
and log-determinants (3.4) are all built from complex-AD primitives
(matrix product, sum, Hermitian transpose, matrix inverse / solve,
Cholesky, $\log\det$). PyTorch's complex autograd composes the
Wirtinger calculus through every step, so any
$I(V_A; V_B \mid V_C)$ is an end-to-end differentiable function of the
design parameter $\eta$, and a single reverse-mode AD sweep delivers
$\nabla_{\eta^*} I$ at every controllable factor with no manual
derivation.

---

## 4. Conditional-MI objectives

A physical-layer design objective is seldom a single conditional MI.
The library targets the following hierarchy.

### 4.1 Linear conditional-MI objectives

A weighted aggregate of conditional mutual informations,

$$U(\eta) = \sum_{n=1}^{N} \alpha_n \, I(V_{A_n}; V_{B_n} \mid V_{C_n}), \tag{4.1}$$

with positive weights $\alpha_n$ encodes a weighted sum-rate or a
fairness criterion. By the Proposition of §3, each summand shares the
*same* K-recursion graph, so $U$ is differentiable in $\eta$ and its
gradient is delivered by a single reverse-mode sweep.

### 4.2 General conditional-MI objectives

More generally, for any differentiable $\Phi : \mathbb{R}^N \to \mathbb{R}$
and any $N$ conditional mutual informations
$I_1, \ldots, I_N$ on $\mathcal{G}$,

$$U(\eta) = \Phi( I_1(\eta), \ldots, I_N(\eta) ), \qquad I_n(\eta) := I(V_{A_n}; V_{B_n} \mid V_{C_n}), \tag{4.2}$$

defines a **conditional-MI objective**. The chain rule makes $U$
differentiable in $\eta$ regardless of $N$, of the choice of $\Phi$,
or of the DAG topology, and a single reverse-mode AD sweep returns
$\nabla_{\eta^*} U$ for all controllable factors at once. The linear
objectives (4.1) are the special case of linear $\Phi$; the rate-region
functions of §5 are real-linear $\Phi$'s; the composite sigmoid
surrogate of §6 is a non-linear $\Phi$.

---

## 5. Multi-terminal rate-region functions

For $K$ user rates $R = (R_1, \ldots, R_K)$, the achievable region at
design $\eta$ is an intersection of $\log\det$ inequalities,

$$\mathcal{R}(\eta) = \{ R \in \mathbb{R}_+^K : \sum_{k \in T} R_k \leq f_T(\eta), \;\; \forall T \in \mathcal{S} \}, \tag{5.1}$$

indexed by a family $\mathcal{S} \subseteq 2^{[K]} \setminus \{\emptyset\}$.
Each **rate function** is a real-linear combination of conditional
mutual informations,

$$f_T(\eta) = \sum_{n=1}^{N_T} \alpha_{T, n} \, I(V_{A_{T, n}}; V_{B_{T, n}} \mid V_{C_{T, n}}), \tag{5.2}$$

with coefficients $\alpha_{T, n} \in \mathbb{R}$ (either sign allowed
for relay bounds and Han–Kobayashi terms). Familiar regions are special
cases:

- **Two-user MAC pentagon** ($N_T = 1$):
  $f_T(\eta) = I(V_{X_T}; V_Y \mid V_{X_{T^c}})$.
- **Gaussian broadcast with dirty-paper coding** (Caire–Shamai).
- **Han–Kobayashi inner bound** for the interference channel
  ($N_T \geq 2$, mixed-sign coefficients).
- **Decode- and compress-and-forward relay inner bounds**
  (Cover–El Gamal).

All rate functions are evaluated by `evaluate_rate_functions` in
`cmi_dag.rate_region` from the same K-blocks.

**Example (two-user MAC).** Two rate-bearing roots $X_1, X_2$ feed a
single receiver $Y$. Three conditional MIs generate the standard
objectives:

- interference-free user rates
  $I_1 = I(V_{X_1}; V_Y \mid V_{X_2})$,
  $I_2 = I(V_{X_2}; V_Y \mid V_{X_1})$,
- sum information $I_{12} = I(V_{X_1 X_2}; V_Y)$.

Then:

- *(i)* sum throughput: $U = I_{12}$;
- *(ii)* weighted sum-rate: $U = \alpha_1 I_1 + \alpha_2 I_2$;
- *(iii)* proportional fairness: $U = \log I_1 + \log I_2$ (non-linear);
- *(iv)* MAC-pentagon composite outage surrogate (see §6).

All four read off the one K-recursion graph and differ only in $\Phi$.

---

## 6. Composite sigmoid surrogate of rate-region outage

A useful **non-linear** conditional-MI objective is built from the rate
functions themselves. At an operating point $R$, the achievability
indicator factors as
$\mathbf{1}\{R \in \mathcal{R}(\eta)\} = \prod_{T \in \mathcal{S}} \mathbf{1}\{\sum_{k \in T} R_k \leq f_T(\eta)\}$.
Smoothing each indicator by a temperature-$\tau$ sigmoid yields the
**composite sigmoid surrogate** of the rate-region outage indicator,

$$\widehat{\rho}_\tau(\eta) = 1 - \prod_{T \in \mathcal{S}} \sigma_\tau( f_T(\eta) - \sum_{k \in T} R_k ). \tag{6.1}$$

This is an instance of (4.2) whose $\Phi$ is the non-linear
sigmoid–product composition, with auxiliary constants given by the
target rates $R$ and the temperature $\tau$.

In a fading environment, with the channel realization $H$ an explicit
random argument, the expectation
$\mathbb{E}_H[ \widehat{\rho}_\tau(\eta, H) ]$
is a differentiable proxy for the rate-region outage probability
$\Pr_H[ R \notin \mathcal{R}(\eta, H) ]$, optimized by
mini-batched stochastic PGM with annealed temperature.

---

## 7. Optimization by the projected gradient method

A conditional-MI objective (4.2) is **maximized** when it scores a
communication benefit (a weighted sum-rate, a rate function $f_T$)
and **minimized** when it scores a cost (the composite outage
surrogate (6.1)). Both directions are handled over the feasible set
$\mathcal{F}$ by the projected gradient method (PGM): with step size
$\alpha_t > 0$,

$$\eta^{(t+1)} = \Pi_{\mathcal{F}}( \eta^{(t)} \pm \alpha_t \, \nabla_{\eta^*} U(\eta^{(t)}) ), \tag{7.1}$$

the $+$ sign giving ascent and the $-$ sign descent. $\Pi_{\mathcal{F}}$
is the Euclidean projection onto $\mathcal{F}$ — in closed form for
the constraint sets used (per-user Frobenius balls, shared total
power, unit-modulus). Each iteration is one K-recursion forward pass,
one reverse-mode AD backward pass, and one closed-form projection;
the gradient is exact up to the Wirtinger convention.

Implementation: `pga_ascent` (maximize) and `pga_descent` (minimize)
in `cmi_dag.optimize`.

As $U$ is in general non-convex in $\eta$, the iterates of (7.1)
converge to a stationary point rather than a global optimum;
multi-start is recommended for production use.

**Point-to-point special case.** With a single rate-bearing user
($K = 1$), a rate function reduces to one unconditional mutual
information $I(V_X; V_Y)$ and (7.1) degenerates to the exact
information-gradient optimization implemented by
[`gaussian-dag`](https://github.com/wadayama/gaussian-dag). The
conditional-MI objective (4.2) is its multi-terminal generalization.

---

## Notation summary

| Symbol | Meaning |
| --- | --- |
| $V_j \in \mathbb{C}^{d_j}$ | DAG node ($j = 1, \ldots, M$). |
| $\mathcal{R} = \{1, \ldots, K\}$ | User-input roots, one per rate-bearing user. |
| $\mathrm{Pa}(j)$ | Parent set of node $j$. |
| $A_{ji}, A_{ji}^{(\ell)}$ | Edge factor on edge $i \to j$ and its $\ell$-th sub-factor. |
| $\Sigma_r$ | Input covariance at root $r$. |
| $\Sigma_j$ | Independent additive Gaussian noise covariance at node $j$. |
| $K_{jk} = \mathbb{E}[V_j V_k^{\mathsf{H}}]$ | Node-pair covariance block. |
| $A, B, C$ | Disjoint node subsets in a conditional MI. |
| $S = A \cup B \cup C$ | Support of a conditional-MI evaluation. |
| $\Sigma_{X, Y}$ | Block sub-covariance with row set $X$, column set $Y$. |
| $\Sigma_{S, S}$ | Support block covariance (eq. 3.2). |
| $\Sigma_{A \mid X}$ | Conditional covariance of $V_A$ given $V_X$ (Schur). |
| $I(V_A; V_B \mid V_C)$ | Conditional mutual information, in nats. |
| $U(\eta)$ | Conditional-MI objective (4.2). |
| $f_T, \mathcal{R}(\eta)$ | Rate function and achievable rate region. |
| $\widehat{\rho}_\tau$ | Composite sigmoid outage surrogate (6.1). |
| $\sigma_\tau$ | Sigmoid with temperature $\tau$. |
| $\eta$ | Tuple of controllable edge factors (design parameter). |
| $\mathcal{F}, \Pi_{\mathcal{F}}$ | Feasible set and its Euclidean projection (eq. 7.1). |

---

## What's next

The implementation conventions, public API, examples, and tutorials
are described in [`README.md`](README.md) and the walkthroughs under
[`docs/`](docs/). The experimental sections of the accompanying
manuscript (rate-region maximization on a two-user MIMO MAC, secure
precoding on a MIMO wiretap channel, and rate-region maximization on
a randomly generated multi-hop MAC network) are reproduced by the
scripts under [`examples/`](examples/).
