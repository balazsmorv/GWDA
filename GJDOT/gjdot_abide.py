"""ABIDE (Preprocessed Connectomes Project) loading and connectome-graph construction.

Each subject becomes one PyG graph over the atlas ROIs:

``x``
    Node features: row ``i`` of the Fisher-z connectivity matrix (``"corr_row"``)
    or a constant ``1`` (``"ones"``).
``edge_index`` / ``edge_weight``
    The strongest ``edge_top_percent`` % positive correlations, undirected.
``conn``
    The full Fisher-z connectivity matrix, shaped ``(1, N, N)`` so PyG batches it
    without clashing with node-level attributes. Used by :mod:`GJDOT.gjdot_costs`.
``y``
    ``1`` = autism spectrum disorder, ``0`` = typical control (``DX_GROUP`` 1 / 2).
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


ASD_DX_GROUP = 1
NODE_FEATURE_MODES = ("corr_row", "ones")


def fetch_abide(
    data_dir,
    site_ids,
    derivative="rois_cc200",
    pipeline="cpac",
    band_pass_filtering=True,
    global_signal_regression=True,
    quality_checked=True,
):
    """Download (on first use) and load ABIDE PCP ROI time series for ``site_ids``."""
    from nilearn import datasets

    return datasets.fetch_abide_pcp(
        data_dir=str(data_dir),
        derivatives=derivative,
        pipeline=pipeline,
        band_pass_filtering=band_pass_filtering,
        global_signal_regression=global_signal_regression,
        quality_checked=quality_checked,
        SITE_ID=list(site_ids),
        verbose=0,
    )


def fisher_z_connectivity(time_series):
    """Fisher-z transformed Pearson correlation of ``(T, N)`` ROI time series.

    ROIs with a constant signal have undefined correlations; they are set to 0
    (no connectivity). The diagonal is 0.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(np.asarray(time_series, dtype=np.float64), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
    np.fill_diagonal(z, 0.0)
    return z


def node_features(z, mode="corr_row"):
    if mode == "corr_row":
        return z.copy()
    if mode == "ones":
        return np.ones((z.shape[0], 1), dtype=np.float64)
    raise ValueError(f"Unknown node feature mode: {mode!r}; expected one of {NODE_FEATURE_MODES}.")


def top_percent_edges(z, percent):
    """Undirected edges for the strongest ``percent`` % positive off-diagonal correlations."""
    rows, cols = np.triu_indices(z.shape[0], k=1)
    weights = z[rows, cols]
    num_keep = int(round(percent / 100.0 * weights.size))
    order = np.argsort(-weights)[:num_keep]
    order = order[weights[order] > 0]
    rows, cols, weights = rows[order], cols[order], weights[order]
    edge_index = np.vstack([np.concatenate([rows, cols]), np.concatenate([cols, rows])])
    edge_weight = np.concatenate([weights, weights])
    return (
        torch.as_tensor(edge_index, dtype=torch.long),
        torch.as_tensor(edge_weight, dtype=torch.float32),
    )


def build_connectome_graph(time_series, label, site, subject_id, edge_top_percent=10.0, node_feature_mode="corr_row"):
    z = fisher_z_connectivity(time_series)
    edge_index, edge_weight = top_percent_edges(z, edge_top_percent)
    return Data(
        x=torch.as_tensor(node_features(z, node_feature_mode), dtype=torch.float32),
        edge_index=edge_index,
        edge_weight=edge_weight,
        conn=torch.as_tensor(z, dtype=torch.float32).unsqueeze(0),
        y=torch.tensor([label], dtype=torch.long),
        site=site,
        subject_id=subject_id,
    )


def load_abide_graphs(
    data_dir,
    site_ids,
    derivative="rois_cc200",
    edge_top_percent=10.0,
    node_feature_mode="corr_row",
):
    """Connectome graphs for every quality-checked subject of ``site_ids``."""
    abide = fetch_abide(data_dir, site_ids, derivative=derivative)
    phenotypic = pd.DataFrame(abide.phenotypic).reset_index(drop=True)
    time_series = abide[derivative]

    graphs = []
    for idx, ts in enumerate(time_series):
        row = phenotypic.iloc[idx]
        graphs.append(
            build_connectome_graph(
                ts,
                label=int(int(row["DX_GROUP"]) == ASD_DX_GROUP),
                site=str(row["SITE_ID"]),
                subject_id=int(row["SUB_ID"]),
                edge_top_percent=edge_top_percent,
                node_feature_mode=node_feature_mode,
            )
        )
    return graphs
