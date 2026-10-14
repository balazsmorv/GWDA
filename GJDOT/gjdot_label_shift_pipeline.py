"""Pipeline integration of the label-shift correction.

Ported from the gjdot repository (commit 26c5af5), without the per-iteration
re-estimation: ``pi_hat`` is estimated once, before the BCD loop. The
estimators themselves live in :mod:`GJDOT.gjdot_label_shift`; this module
wires them into a pipeline's ``run_experiment``.
"""

import mlflow
import numpy as np

from GJDOT.gjdot_label_shift import (
    class_share,
    estimate_target_proportions,
    ot_cost_at,
    ot_proportion_profile,
)
from GJDOT.gjdot_logging import log_ot_profile, log_q_trajectory


# Estimators: None (uniform source marginal) | "oracle" | "ot".
def q_grid(params):
    """Grid of class-1 shares q searched by the "ot" estimator and the OT-profile diagnostics."""
    return np.linspace(0.0, 1.0, int(params.get("label_shift_q_grid_points", 101)))


def label_shift_method(params):
    """Short name of the marginal variant: "uniform" or the estimator name."""
    return params.get("label_shift_estimator") or "uniform"


# Returns the part of the mlflow run name that shows which label-shift variant was run
def label_shift_run_name_suffix(params):
    if not params.get("label_shift_estimator"):
        return ""
    return f" marginal={label_shift_method(params)}"


def _fit_label_shift_estimator(method, source_graphs, cost, oracle_labels, grid, num_classes=2):
    """pi_hat of the target given `source_graphs`; `cost` is the (n_s, n_t) graph cost to the target."""
    ys = np.array([int(g.y.item()) for g in source_graphs], dtype=np.int64)
    if method == "oracle":
        return estimate_target_proportions("oracle", y_target=oracle_labels, num_classes=num_classes)
    if method == "ot":
        return estimate_target_proportions("ot", ys=ys, D=cost, grid=grid)
    raise ValueError(f"Unknown label_shift_estimator {method!r}")


def _run_target_label_estimation(params, source_graphs, target_graphs, cost_mtx, num_classes=2):
    """Estimate pi_hat on the target graphs and log label-shift diagnostics."""

    method = params["label_shift_estimator"]
    binary = num_classes == 2

    ys = np.array([int(g.y.item()) for g in source_graphs], dtype=np.int64)
    yt = np.array([int(g.y.item()) for g in target_graphs], dtype=np.int64) # used only for diagnostics

    info = {}
    pi_hat = None

    # compute estimated target label proportions
    if method is not None:
        pi_hat = _fit_label_shift_estimator(
            method, source_graphs, cost_mtx, yt, q_grid(params), num_classes=num_classes
        )

    # log class ratios and errors for binary tasks
    if binary:
        info["q_source"] = class_share(ys)
        info["q_star"] = class_share(yt)
        # A uniform source marginal implicitly assumes the source class prior.
        info["q_hat"] = float(pi_hat[1]) if pi_hat is not None else info["q_source"]
        info["abs_error"] = abs(info["q_hat"] - info["q_star"])

    # log OT profile diagnostics (curve depending on q)
    profile = None
    if binary and params["label_shift_diagnostics"]:
        grid, costs = ot_proportion_profile(cost_mtx, ys, grid=q_grid(params))
        profile = {"q": grid.tolist(), "ot_cost": costs.tolist()}
        info["ot_profile_argmin_q"] = float(grid[int(np.argmin(costs))])
        info["ot_profile_min_cost"] = float(np.min(costs))
        info["residual"] = ot_cost_at(cost_mtx, ys, info["q_hat"])
        info["ot_cost_at_q_star"] = ot_cost_at(cost_mtx, ys, info["q_star"])
        info["ot_cost_at_q_source"] = ot_cost_at(cost_mtx, ys, info["q_source"])

    for key, value in info.items():
        mlflow.log_metric(f"label_shift_{key}", float(value))
    if pi_hat is not None:
        mlflow.log_dict({"pi_hat": pi_hat.tolist()}, "label_shift/pi_hat.json")
    if profile is not None:
        mlflow.log_dict(profile, "label_shift/ot_profile.json")
        markers = [("true q*", info["q_star"], "-"), ("source q", info["q_source"], ":")]
        if method is not None:
            markers.append((f"{method} q_hat", info["q_hat"], "--"))
        log_ot_profile(
            profile["q"],
            profile["ot_cost"],
            markers,
            f"{params.get('source_domain')} → {params.get('target_domain')}: OT profile (dot = minimum)",
            "label_shift/ot_profile.png",
        )
    return pi_hat, info


def estimate_target_label_proportions(params, *, source_graphs, target_graphs, cost_mtx, num_classes=2):

    # Check settings
    estimator = params.get("label_shift_estimator")
    if estimator not in (None, "oracle") and num_classes != 2:
        raise ValueError(f"label_shift_estimator={estimator!r} is implemented for binary tasks only.")

    # When there's no estimator and diagnostics are off, it returns the default
    if estimator is None and not params.get("label_shift_diagnostics", False):
        return None, {}

    # Estimate target label proportions and log the diagnostics
    return _run_target_label_estimation(params, source_graphs, target_graphs, cost_mtx, num_classes=num_classes)


def log_label_shift_trajectory(params, ys, yt, q_marginal, q_predicted):
    """Log how the class-1 share evolves over the BCD iterations (one plot per run, binary tasks).

    `q_marginal[k]` is the class-1 mass of the source marginal used at iteration k (constant,
    since pi_hat is estimated once); `q_predicted[k]` is the share of target graphs the
    classifier predicts as class 1 after iteration k. Index 0 is the state before BCD
    (initial marginal, source-only baseline).
    """
    steps = list(range(len(q_marginal)))
    q_star, q_source = class_share(yt), class_share(ys)
    mlflow.log_dict({"bcd_iteration": steps, "q_marginal": q_marginal, "q_predicted": q_predicted,
                     "q_star": q_star, "q_source": q_source}, "label_shift/q_trajectory.json")
    log_q_trajectory(
        steps,
        [(f"source marginal ({label_shift_method(params)})", q_marginal, "#2a78d6", "-"),
         ("classifier prediction on target", q_predicted, "#eb6834", "--")],
        [("true q*", q_star, "-"), ("source q", q_source, ":")],
        f"{params.get('source_domain')} → {params.get('target_domain')}: class-1 share per BCD iteration",
        "label_shift/q_trajectory.png",
    )
