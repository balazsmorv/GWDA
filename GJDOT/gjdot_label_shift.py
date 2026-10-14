"""Label-shift correction for the GJDOT outer coupling.

Ported from the gjdot repository (commit 26c5af5).

The outer coupling's source marginal is uniform by default,
which implicitly assumes the target class proportions equal the source ones.

``source_marginal(ys, pi_hat=...)`` instead puts mass
``pi_hat[y_i] / n_{y_i}`` on each source sample, so the transported source
class mix matches an estimate ``pi_hat`` of the target class proportions.

``estimate_target_proportions`` provides that estimate from the (unlabelled)
target graphs the GJDOT run adapts to; only ``"oracle"`` reads target labels.

Binary estimators return ``pi_hat = [1 - q, q]`` with ``q`` the class-1 share.
"""

import numpy as np
import ot


LABEL_SHIFT_ESTIMATORS = ("oracle", "ot")
DEFAULT_Q_GRID = np.linspace(0.0, 1.0, 101)


# ---------------------------------------------------------------------------
# Source marginal
# ---------------------------------------------------------------------------
def source_marginal(ys, pi_hat=None, num_classes=2):
    """Uniform ``1/n_s`` when ``pi_hat`` is None, else ``pi_hat[y_i] / n_{y_i}``."""
    ys = np.asarray(ys).reshape(-1).astype(np.int64)
    if pi_hat is None:
        return ot.unif(len(ys))
    pi_hat = np.asarray(pi_hat, dtype=np.float64)
    counts = np.bincount(ys, minlength=num_classes).astype(np.float64)
    if np.any(counts[pi_hat > 0] == 0):
        raise ValueError("pi_hat puts mass on a class with no source samples.")
    return pi_hat[ys] / np.maximum(counts[ys], 1.0)


def binary_source_marginal(ys, q):
    return source_marginal(ys, pi_hat=np.array([1.0 - q, q]), num_classes=2)


def class_share(y, positive=1):
    y = np.asarray(y).reshape(-1)
    return float(np.mean(y == positive)) if y.size else float("nan")


# ---------------------------------------------------------------------------
# OT profile
# ---------------------------------------------------------------------------
def ot_proportion_profile(D, ys, grid=DEFAULT_Q_GRID, num_itermax=1_000_000):
    """``cost(q) = emd2(a(q), unif, D)`` for every ``q`` in ``grid``."""
    D = np.asarray(D, dtype=np.float64)
    b = ot.unif(D.shape[1])
    costs = np.array(
        [ot.emd2(binary_source_marginal(ys, q), b, D, numItermax=num_itermax) for q in grid],
        dtype=np.float64,
    )
    return np.asarray(grid, dtype=np.float64), costs


def ot_cost_at(D, ys, q, num_itermax=1_000_000):
    D = np.asarray(D, dtype=np.float64)
    return float(ot.emd2(binary_source_marginal(ys, q), ot.unif(D.shape[1]), D, numItermax=num_itermax))


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def _pi_from_q(q):
    return np.array([1.0 - q, q], dtype=np.float64)


def _estimate_ot(D, ys, grid=DEFAULT_Q_GRID):
    grid, costs = ot_proportion_profile(D, ys, grid=grid)
    return float(grid[int(np.argmin(costs))])


def estimate_target_proportions(
    method,
    *,
    ys=None,
    D=None,
    y_target=None,
    grid=DEFAULT_Q_GRID,
    num_classes=2,
):
    """Estimate target class proportions ``pi_hat``.
    method:
      ``"ot"``     binary; ``argmin_q emd2(a(q), unif, D)`` over ``grid``. Needs ``ys``, ``D`` (n_s, n_t).
      ``"oracle"`` true proportions from ``y_target`` (diagnostic only).
    """
    if method == "oracle":
        y_target = np.asarray(y_target).reshape(-1).astype(np.int64)
        counts = np.bincount(y_target, minlength=num_classes).astype(np.float64)
        return counts / counts.sum()

    if num_classes != 2:
        raise ValueError(f"estimator {method!r} is implemented for binary tasks only.")
    if method == "ot":
        return _pi_from_q(_estimate_ot(D, ys, grid=grid))
    raise ValueError(f"Unknown label-shift estimator {method!r}; expected one of {LABEL_SHIFT_ESTIMATORS}.")
