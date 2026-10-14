"""Graph-level ground costs between source and target connectome graphs.

GJDOT needs one scalar cost per (source graph, target graph) pair. Two costs are
supported, selected by ``cost_type``:

``fgw``
    Fused Gromov-Wasserstein distance (POT). The structure matrix of a graph is
    its correlation distance ``1 - r`` and the feature cost is the mean squared
    Euclidean distance between node features, so both FGW terms live on a
    comparable scale. ``fgw_alpha`` weights the structure term, as in POT.

``frobenius``
    Frobenius norm ``||Z_s - Z_t||_F`` between the Fisher-z connectivity
    matrices. Every subject is parcellated with the same atlas, so node ``i`` is
    the same brain region in every graph and no node matching is needed.

Both read the connectivity matrix stored on each graph as ``graph.conn`` (see
:mod:`GJDOT.gjdot_abide`).
"""

import numpy as np
import ot
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist


COST_TYPES = ("fgw", "frobenius")
GRAPH_COST_NORMALIZATIONS = (None, "max", "minmax")


def connectivity_matrix(graph):
    """Fisher-z connectivity matrix ``(N, N)``; stored as ``(1, N, N)`` so PyG batches it."""
    return graph.conn[0].detach().cpu().numpy().astype(np.float64)


def correlation_distance_matrix(graph):
    """FGW structure matrix: correlation distance ``1 - r`` with a zero diagonal."""
    distance = 1.0 - np.tanh(connectivity_matrix(graph))
    np.fill_diagonal(distance, 0.0)
    return distance


def node_feature_matrix(graph):
    return graph.x.detach().cpu().numpy().astype(np.float64)


def frobenius_cost_matrix(source_graphs, target_graphs):
    source = np.stack([connectivity_matrix(graph).ravel() for graph in source_graphs])
    target = np.stack([connectivity_matrix(graph).ravel() for graph in target_graphs])
    return cdist(source, target, metric="euclidean")


def _fgw_distance(C1, C2, F1, F2, alpha, max_iter=500, tol_abs=1e-5):
    M = cdist(F1, F2, metric="sqeuclidean") / F1.shape[1]
    p = ot.unif(C1.shape[0])
    q = ot.unif(C2.shape[0])
    return float(
        ot.gromov.fused_gromov_wasserstein2(
            M,
            C1,
            C2,
            p,
            q,
            loss_fun="square_loss",
            alpha=alpha,
            max_iter=max_iter,
            tol_abs=tol_abs,
            log=False,
            verbose=False,
        )
    )


def _fgw_cost_row(i, n_source, C1, F1, target_structures, target_features, alpha):
    print(i, "/", n_source, flush=True)
    return [
        _fgw_distance(C1, C2, F1, F2, alpha)
        for C2, F2 in zip(target_structures, target_features)
    ]


def fgw_cost_matrix(source_graphs, target_graphs, alpha=0.5, n_jobs=1):
    source_structures = [correlation_distance_matrix(graph) for graph in source_graphs]
    target_structures = [correlation_distance_matrix(graph) for graph in target_graphs]
    source_features = [node_feature_matrix(graph) for graph in source_graphs]
    target_features = [node_feature_matrix(graph) for graph in target_graphs]

    rows = Parallel(n_jobs=n_jobs)(
        delayed(_fgw_cost_row)(i, len(source_graphs), C1, F1, target_structures, target_features, alpha)
        for i, (C1, F1) in enumerate(zip(source_structures, source_features))
    )
    return np.asarray(rows, dtype=np.float64).reshape(len(source_graphs), len(target_graphs))


def compute_cost_matrix(source_graphs, target_graphs, cost_type, fgw_alpha=0.5, n_jobs=1):
    if cost_type == "fgw":
        return fgw_cost_matrix(source_graphs, target_graphs, alpha=fgw_alpha, n_jobs=n_jobs)
    if cost_type == "frobenius":
        return frobenius_cost_matrix(source_graphs, target_graphs)
    raise ValueError(f"Unknown cost_type: {cost_type!r}; expected one of {COST_TYPES}.")


def normalize_graph_cost(cost, mode, eps=1e-12):
    """Rescale a graph cost matrix so ``jdot_alpha`` means the same for every cost type.

    ``"max"`` divides by the maximum; ``"minmax"`` maps the costs onto ``[0, 1]``,
    so the whole spread of the graph cost competes with the label loss.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if mode is None:
        return cost
    if mode == "max":
        return cost / max(float(np.max(cost)), eps)
    if mode == "minmax":
        low = float(np.min(cost))
        return (cost - low) / max(float(np.max(cost)) - low, eps)
    raise ValueError(f"Unknown graph cost normalization: {mode!r}; expected one of {GRAPH_COST_NORMALIZATIONS}.")


def load_or_compute_cost_matrix(
    cost_path,
    source_graphs,
    target_graphs,
    cost_type,
    fgw_alpha=0.5,
    n_jobs=1,
    reload=False,
    save=True,
):
    if cost_path.exists() and not reload:
        print(f"Loading cached cost matrix from {cost_path}")
        return np.load(cost_path)["cost_mtx"]

    cost_mtx = compute_cost_matrix(
        source_graphs,
        target_graphs,
        cost_type=cost_type,
        fgw_alpha=fgw_alpha,
        n_jobs=n_jobs,
    )

    if save:
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cost_path, cost_mtx=cost_mtx)
        print(f"Saved cost matrix to {cost_path}")
    return cost_mtx
