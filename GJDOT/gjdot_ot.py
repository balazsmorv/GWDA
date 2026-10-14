"""OT and JDOT-objective helpers for the GJDOT block-coordinate-descent loop.

Ported from the gjdot repository (commit 26c0075). The graph-level cost
matrices (FGW / Frobenius) live in :mod:`GJDOT.gjdot_costs`.
"""

import warnings

import numpy as np
import ot
import torch
import torch.nn.functional as F


def build_oracle_label_loss_matrix(source_graphs, target_graphs):
    source_y = np.array([int(g.y.item()) for g in source_graphs], dtype=np.int64).reshape(-1, 1)
    target_y = np.array([int(g.y.item()) for g in target_graphs], dtype=np.int64).reshape(1, -1)
    return (source_y != target_y).astype(np.float64)


def _collapse_label_loss(loss):
    if loss.ndim > 1 and loss.shape[-1] > 1:
        return loss.sum(dim=-1)
    if loss.ndim > 1 and loss.shape[-1] == 1:
        return loss.squeeze(-1)
    return loss


def label_loss_per_sample(y_pred, y_true, label_loss="cross_entropy"):
    y_true = y_true.clamp(0.0, 1.0)

    if label_loss == "squared_error":
        return _collapse_label_loss((y_pred - y_true).pow(2))

    if label_loss == "cross_entropy":
        y_pred = y_pred.clamp(1e-12, 1.0 - 1e-12)
        if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
            return -(y_true * torch.log(y_pred)).sum(dim=-1)
        return _collapse_label_loss(F.binary_cross_entropy(y_pred, y_true, reduction="none"))

    if label_loss == "squared_hinge":
        positive_loss = torch.relu(1.0 - y_pred).pow(2)
        negative_loss = torch.relu(1.0 + y_pred).pow(2)
        return _collapse_label_loss(y_true * positive_loss + (1.0 - y_true) * negative_loss)

    raise ValueError(f"Unknown label_loss: {label_loss}")


def label_loss(y_source, y_pred, label_loss_type="cross_entropy"):
    y_source = torch.as_tensor(y_source, dtype=torch.float64)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float64)
    if y_source.ndim == 1:
        y_source = y_source.view(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.view(-1, 1)

    pairwise_loss = label_loss_per_sample(
        y_pred[None, :, :].expand(y_source.shape[0], -1, -1),
        y_source[:, None, :].expand(-1, y_pred.shape[0], -1),
        label_loss=label_loss_type,
    )
    return pairwise_loss.cpu().numpy()


def compute_jdot_terms(gamma, graph_costs, label_loss_matrix, jdot_alpha):
    gamma = np.asarray(gamma, dtype=np.float64)
    jdot_objective_graph_part = float(
        np.sum(gamma * (float(jdot_alpha) * np.asarray(graph_costs, dtype=np.float64)))
    )
    jdot_objective_label_loss_part = float(
        np.sum(gamma * np.asarray(label_loss_matrix, dtype=np.float64))
    )
    return {
        "jdot_objective": jdot_objective_graph_part + jdot_objective_label_loss_part,
        "jdot_objective_graph_part": jdot_objective_graph_part,
        "jdot_objective_label_loss_part": jdot_objective_label_loss_part,
    }


def compute_jdot_alpha(value, cost_mtx):
    if value is None:
        return 1.0 / max(float(np.max(cost_mtx)), 1e-12)
    return float(value)


def normalize_ot_cost(cost, eps=1e-12):
    cost = np.asarray(cost, dtype=np.float64)
    finite = cost[np.isfinite(cost)]
    scale = max(float(np.max(finite)), float(eps)) if finite.size else float(eps)
    return cost / scale, scale

def solve_ot(params, w_s, w_t, cost, ys=None, numItermax=100000, stopThr=1e-5):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Warning)
            if params.get("use_label_reg", False):
                return ot.da.sinkhorn_l1l2_gl(
                    w_s,
                    np.ravel(ys),
                    w_t,
                    cost,
                    reg=params["label_reg"],
                )
            if params.get("use_entropy_reg", False):
                return ot.sinkhorn(
                    w_s,
                    w_t,
                    cost,
                    reg=params["entropy_reg"],
                    method="sinkhorn",
                    numItermax=numItermax,
                    stopThr=stopThr,
                    verbose=False,
                )
            if params.get("use_unbalanced", False):
                return ot.unbalanced.sinkhorn_unbalanced(
                    w_s,
                    w_t,
                    cost,
                    reg=params["entropy_reg"],
                    reg_m=params["unbalanced_reg"] if not params.get("semi-relaxed", False) else (params["unbalanced_reg"], float("inf")),
                    reg_type=params.get("unbalanced_reg_type", "kl"),
                    numItermax=numItermax,
                    stopThr=stopThr,
                )
            return ot.emd(w_s, w_t, cost, numItermax=numItermax)
    except Warning as warning:
        print(f"Skipping OT solve because POT emitted warning: {warning}")
        return None


def swapped_domain_params(params):
    swapped = dict(params)
    swapped["source_domain"] = params["target_domain"]
    swapped["target_domain"] = params["source_domain"]
    return swapped
