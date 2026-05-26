"""
Association measures for 2x2 contingency tables.

Provides three association measures used in the corpus-linguistics literature,
designed to be applied to (paper-level) co-occurrence counts:

    LR   - relative risk (a.k.a. risk ratio): P(B|A) / P(B|~A).
           This is what the paper has historically called "Likelihood Ratio".
           Strictly speaking it is the *relative risk* from epidemiology; the
           *statistical* likelihood-ratio test is G^2 (see below).

           Bounded in [-1, 1]. 0 = independence, 1 = perfect co-occurrence,
           -1 = never co-occur. Symmetric in A and B.

    G^2  - Dunning log-likelihood (Dunning 1993). The standard
           likelihood-ratio test for independence in a 2x2 table. Asymptotically
           chi^2_1 under the null hypothesis of independence. Combines effect
           size and sample size in one number; well-behaved at low counts where

A 2x2 contingency table is parameterised as:

                 B          ~B
       A     k11 (a)    k12 (b)    n1. = a+b
      ~A     k21 (c)    k22 (d)    n2. = c+d
             n.1 = a+c  n.2 = b+d  N   = a+b+c+d

The functions below accept either scalar counts or 1-D numpy arrays of
counts (one entry per pair), and return scalar floats or numpy arrays.

Multiple-testing correction:
    bh_fdr(pvals) returns Benjamini-Hochberg adjusted q-values.

References
----------
    extraction. Proceedings of GSCL.
Dunning, T. (1993). Accurate methods for the statistics of surprise and
    coincidence. Computational Linguistics, 19(1), 61-74.
Manning, C. D., & Schuetze, H. (1999). Foundations of Statistical Natural
    Language Processing, Ch. 5.
Evert, S. (2008). Corpora and collocations. In Luedeling & Kyto (eds.),
    Corpus Linguistics: An International Handbook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

try:
    from scipy.stats import chi2

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is in the existing requirements
    _HAS_SCIPY = False


__all__ = [
    "PairCounts",
    "relative_risk",
    "g2",
    "g2_pvalue",
    "bh_fdr",
    "compute_all",
]


@dataclass(frozen=True)
class PairCounts:
    """Convenience container for a 2x2 contingency table."""

    k11: int  # papers with A and B
    k12: int  # papers with A but not B
    k21: int  # papers with B but not A
    k22: int  # papers with neither A nor B

    @classmethod
    def from_marginals(cls, k11: int, n_a: int, n_b: int, n_total: int) -> "PairCounts":
        """Build a PairCounts from the joint count and marginal counts.

        n_a    = total papers with A
        n_b    = total papers with B
        n_total = total papers in the corpus
        """
        k12 = n_a - k11
        k21 = n_b - k11
        k22 = n_total - n_a - n_b + k11
        return cls(k11=k11, k12=k12, k21=k21, k22=k22)


def _as_array(*xs: int | float | np.ndarray) -> tuple[np.ndarray, ...]:
    arrs = [np.asarray(x, dtype=float) for x in xs]
    return tuple(np.broadcast_arrays(*arrs))


def relative_risk(
    k11: int | np.ndarray,
    k12: int | np.ndarray,
    k21: int | np.ndarray,
    k22: int | np.ndarray,
    *,
    add_one: bool = False,
) -> float | np.ndarray:
    """Relative risk (the paper's "LR"): P(B|A) / P(B|~A).

    If add_one=True, applies a +1 Laplace smoothing to all four cells
    (useful when k21=0 produces +inf). The default mirrors the existing
    pipeline (no smoothing): the result is +inf when the denominator is 0
    but the numerator is positive, and 1.0 (independence) when both are 0.
    """
    a, b, c, d = _as_array(k11, k12, k21, k22)
    if add_one:
        a, b, c, d = a + 1, b + 1, c + 1, d + 1
    n_a = a + b
    n_not_a = c + d
    p_b_given_a = np.divide(a, n_a, out=np.zeros_like(a), where=n_a > 0)
    p_b_given_not_a = np.divide(c, n_not_a, out=np.zeros_like(c), where=n_not_a > 0)

    out = np.where(
        p_b_given_not_a > 0,
        p_b_given_a / np.where(p_b_given_not_a == 0, 1.0, p_b_given_not_a),
        np.where(p_b_given_a > 0, np.inf, 1.0),
    )
    if out.ndim == 0:
        return float(out)
    return out



def _xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x * log(y), with the convention 0 * log(0) = 0."""
    out = np.zeros_like(x, dtype=float)
    mask = (x > 0) & (y > 0)
    out[mask] = x[mask] * np.log(y[mask] / 1.0)
    return out


def g2(
    k11: int | np.ndarray,
    k12: int | np.ndarray,
    k21: int | np.ndarray,
    k22: int | np.ndarray,
) -> float | np.ndarray:
    """Dunning log-likelihood (G^2).

    G^2 = 2 * sum_{i,j} k_{ij} * log(k_{ij} / e_{ij})

    where e_{ij} is the expected count under independence. Asymptotically
    chi-squared with 1 degree of freedom under the null. Always >= 0.

    Empty cells contribute 0 (via the convention 0 * log(0) = 0).
    """
    a, b, c, d = _as_array(k11, k12, k21, k22)
    n = a + b + c + d
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d

    with np.errstate(divide="ignore", invalid="ignore"):
        e11 = row1 * col1 / n
        e12 = row1 * col2 / n
        e21 = row2 * col1 / n
        e22 = row2 * col2 / n

        # Compute k * log(k / e) safely. When k == 0, the term is 0.
        # When e == 0, it must be that the corresponding k is also 0 (because
        # any zero marginal forces e to 0 in the same row/column), so the term
        # is 0; we guard against k > 0 with e == 0 (numerically impossible)
        # by masking.
        def term(k, e):
            ratio = np.where((k > 0) & (e > 0), k / np.where(e == 0, 1.0, e), 1.0)
            return np.where((k > 0) & (e > 0), k * np.log(ratio), 0.0)

        out = 2.0 * (term(a, e11) + term(b, e12) + term(c, e21) + term(d, e22))
        # If the entire table is empty or only one row/column has counts, G^2 = 0.
        out = np.where(n == 0, 0.0, out)

    if out.ndim == 0:
        return float(out)
    return out


def g2_pvalue(g2_value: float | np.ndarray) -> float | np.ndarray:
    """Two-sided p-value for G^2 under the chi^2_1 reference distribution."""
    if not _HAS_SCIPY:  # pragma: no cover
        raise ImportError("scipy is required for g2_pvalue; install scipy")
    g2_arr = np.asarray(g2_value, dtype=float)
    p = chi2.sf(g2_arr, df=1)
    if p.ndim == 0:
        return float(p)
    return p


def bh_fdr(pvals: Iterable[float], *, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals
        Iterable of p-values.
    q
        Target false discovery rate (used only for the boolean rejection mask).

    Returns
    -------
    (qvals, reject)
        qvals  : adjusted p-values (BH q-values), same shape as input.
        reject : boolean mask, True where qvals <= q (i.e. rejected at level q).
    """
    p = np.asarray(list(pvals), dtype=float)
    n = p.size
    if n == 0:
        return p.copy(), np.zeros(0, dtype=bool)

    order = np.argsort(p)
    ranked = p[order]
    # Standard BH adjustment: q_i = min_{j >= i} ( n / j * p_(j) ), enforced monotone.
    bh = ranked * n / (np.arange(n) + 1)
    # Enforce monotone non-increasing from the right.
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)

    qvals = np.empty(n, dtype=float)
    qvals[order] = bh
    reject = qvals <= q
    return qvals, reject


def compute_all(
    k11: int | np.ndarray,
    k12: int | np.ndarray,
    k21: int | np.ndarray,
    k22: int | np.ndarray,
    *,
    fdr_q: float = 0.05,
) -> dict:
    """Compute LR, G^2, p-value, and BH q-value in one pass.

    Returns a dict mapping name -> scalar or numpy array (matching the input
    shape). Useful for vectorised computation over many pairs at once.
    """
    lr = relative_risk(k11, k12, k21, k22)
    g2_val = g2(k11, k12, k21, k22)
    p = g2_pvalue(g2_val)
    p_arr = np.atleast_1d(np.asarray(p, dtype=float))
    qvals, reject = bh_fdr(p_arr, q=fdr_q)
    if np.isscalar(p):
        qvals = float(qvals[0])
        reject = bool(reject[0])
    elif p_arr.shape != np.asarray(p).shape:
        qvals = qvals.reshape(np.asarray(p).shape)
        reject = reject.reshape(np.asarray(p).shape)

    return {
        "lr": lr,
        "g2": g2_val,
        "p_value": p,
        "q_value": qvals,
        "reject": reject,
    }



def _selftest() -> None:
    """A handful of sanity checks against textbook values and edge cases."""
    # 1. Independence: LR = 1, G^2 = 0.
    pc = PairCounts.from_marginals(k11=10, n_a=20, n_b=50, n_total=100)
    assert abs(relative_risk(pc.k11, pc.k12, pc.k21, pc.k22) - 1.0) < 1e-9
    assert abs(g2(pc.k11, pc.k12, pc.k21, pc.k22) - 0.0) < 1e-9

    # 2. Perfect co-occurrence (A iff B): LR = inf.
    pc = PairCounts.from_marginals(k11=10, n_a=10, n_b=10, n_total=100)
    assert relative_risk(pc.k11, pc.k12, pc.k21, pc.k22) == np.inf

    # 3. Strong association (BLEU-MT-style): LR > 1, G^2 large.
    pc = PairCounts.from_marginals(k11=600, n_a=847, n_b=2384, n_total=14171)
    lr = relative_risk(pc.k11, pc.k12, pc.k21, pc.k22)
    g2v = g2(pc.k11, pc.k12, pc.k21, pc.k22)
    p = g2_pvalue(g2v)
    assert lr > 4 and lr < 6, f"LR out of expected range: {lr}"
    assert g2v > 1000, f"G^2 out of expected range: {g2v}"
    assert p < 1e-50, f"p-value out of expected range: {p}"

    # 4. Weak-evidence pair (low joint count, but high RR):
    pc = PairCounts.from_marginals(k11=3, n_a=30, n_b=40, n_total=14171)
    lr = relative_risk(pc.k11, pc.k12, pc.k21, pc.k22)
    g2v = g2(pc.k11, pc.k12, pc.k21, pc.k22)
    p = g2_pvalue(g2v)
    assert lr > 30, f"LR out of expected range: {lr}"
    assert g2v < 50, f"G^2 out of expected range: {g2v}"
    # The point: G^2 flags this as much weaker evidence than the RR magnitude
    # would suggest.

    # 5. Vectorisation: scalar and array calls agree.
    ks = np.array([[600, 247, 1784, 11540], [3, 27, 37, 14104], [10, 10, 40, 40]])
    lr_arr = relative_risk(ks[:, 0], ks[:, 1], ks[:, 2], ks[:, 3])
    g2_arr = g2(ks[:, 0], ks[:, 1], ks[:, 2], ks[:, 3])
    p_arr = g2_pvalue(g2_arr)
    for i in range(ks.shape[0]):
        assert abs(
            lr_arr[i] - relative_risk(ks[i, 0], ks[i, 1], ks[i, 2], ks[i, 3])
        ) < 1e-9
        assert abs(g2_arr[i] - g2(ks[i, 0], ks[i, 1], ks[i, 2], ks[i, 3])) < 1e-9
        assert abs(p_arr[i] - g2_pvalue(g2_arr[i])) < 1e-12

    # 6. BH-FDR sanity: known small example.
    pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205])
    qvals, _ = bh_fdr(pvals)
    # Q-values must be monotone non-decreasing in original p order rank.
    order = np.argsort(pvals)
    ranked_q = qvals[order]
    assert np.all(np.diff(ranked_q) >= -1e-12)
    # The smallest q must equal n*p_(1), capped at 1.
    n = len(pvals)
    assert abs(ranked_q[0] - min(1.0, n * pvals[order][0])) < 1e-12

    # 7. compute_all consistency.
    out = compute_all(np.array([600, 3]), np.array([247, 27]), np.array([1784, 37]), np.array([11540, 14104]))
    assert out["lr"][0] > 4 and out["g2"][0] > 1000
    assert out["lr"][1] > 30 and out["g2"][1] < 50

    print("association_measures.py self-test passed.")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
