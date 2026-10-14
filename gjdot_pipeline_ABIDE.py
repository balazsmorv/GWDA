"""GJDOT domain adaptation between ABIDE acquisition sites (ASD vs. control).

Every grid combination is one MLflow run: a source-only GIN baseline, followed by
the GJDOT block-coordinate descent that alternates an OT solve on
``jdot_alpha * graph_cost + label_loss`` with refitting the GIN on the target
graphs using the OT-transported source labels. The graph cost between subjects is
either the FGW distance or the Frobenius distance of their connectivity matrices
(``COST_TYPES``).
"""

from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import mlflow
import numpy as np
import ot
from torch_geometric.loader import DataLoader

from GJDOT.gjdot_abide import load_abide_graphs
from GJDOT.gjdot_classifiers import (
    classifier_label_loss_metadata,
    make_jdot_classifier,
    resolve_classifier_label_loss,
)
from GJDOT.gjdot_costs import load_or_compute_cost_matrix, normalize_graph_cost
from GJDOT.gjdot_logging import (
    finite_values,
    log_class_label_stats,
    log_confusion_matrix,
    log_histogram,
    log_matrix_image,
    log_ot_marginal_violations,
    log_predicted_class_distribution,
    log_prefixed_metrics,
    log_raw_dataset_stats,
    log_stats,
)
from GJDOT.gjdot_ot import (
    build_oracle_label_loss_matrix,
    compute_jdot_alpha,
    compute_jdot_terms,
    normalize_ot_cost,
    solve_ot,
    swapped_domain_params,
)
from GJDOT.gjdot_utils import (
    iter_grid,
    safe_filename_value,
    select_domain_graphs,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parent
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = f"GJDOT_ABIDE_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
CLASS_LABELS = ["control", "ASD"]  # ABIDE DX_GROUP: 1 = ASD -> 1, 2 = control -> 0

# ABIDE SITE_ID values pooled into one domain.
SITE_GROUPS = {
    "NYU": ["NYU"],
    "UM": ["UM_1", "UM_2"],
    "USM": ["USM"],
    "UCLA": ["UCLA_1", "UCLA_2"],
    "LEUVEN": ["LEUVEN_1", "LEUVEN_2"],
}

# Dataset/domain grid.
DATA_DIRS = [PROJECT_ROOT / "data" / "nilearn_data"]
DERIVATIVES = ["rois_cc200"]  # ABIDE PCP atlas time series (CPAC, band-pass, GSR, quality checked)
EDGE_TOP_PERCENTS = [10.0]  # strongest positive correlations kept as GNN edges
NODE_FEATURE_MODES = ["corr_row"]  # "corr_row" (connectivity profile) or "ones"
SOURCE_DOMAINS = list(SITE_GROUPS)  # all ordered site pairs; source == target is skipped
TARGET_DOMAINS = list(SITE_GROUPS)
SAMPLE_MODES = ["full"]  # "full" or "stratified"
MAX_GRAPHS_PER_CLASS = [None]
SAMPLE_SEEDS = [42]

# Cost matrix grid.
MATRIX_FOLDERS = ["abide"]
COST_TYPES = ["frobenius", "fgw"]  # "frobenius" or "fgw"
FGW_ALPHAS = [0.5]  # only used by cost_type="fgw"
COST_CLIP_PERCENTILES = [None]
GRAPH_COST_NORMALIZATIONS = ["minmax"]  # None, "max" or "minmax"; makes jdot_alpha comparable across cost types
NORMALIZE_OT_COST_VALUES = [True]  # divide the JDOT cost by its max before each OT solve (keeps Sinkhorn stable)
USE_SYMMETRIC_COST_CACHE_VALUES = [True]
DEBUG_ORACLE_LABEL_LOSS = [False]
COST_N_JOBS = -1  # joblib workers for the FGW cost matrix

# Model/training grid.
SEEDS = [42]
IN_DIMS = [None]
HIDDEN_DIMS = [64]
CONV_TYPES = ["gin"]
CONV_LAYERS = [3]
MLP_LAYERS = [2]
DROPOUTS = [0.0]
BATCH_SIZES = [64]
BASELINE_EPOCHS = [100]
JDOT_FIT_EPOCHS = [50]
LRS = [1e-4]
WEIGHT_DECAYS = [0.0]
BASELINE_USE_EARLY_STOPPING_VALUES = [True]
EARLY_STOPPING_PATIENCES = [5]
EARLY_STOPPING_MIN_DELTAS = [0.0]
EARLY_STOPPING_VAL_RATIOS = [0.1]
BCD_USE_EARLY_STOPPING_VALUES = [True]
BCD_EARLY_STOPPING_VAL_RATIOS = [0.1]
BCD_EARLY_STOPPING_PATIENCES = [5]
BCD_EARLY_STOPPING_MIN_DELTAS = [0.0]

# JDOT/OT grid.
BCD_ITERS = [15]
LOG_INSIDE_BCD = [True]
JDOT_ALPHA = [10.0]  # weight of the (normalized) graph cost against the label loss; None = 1 / max(graph cost)

LABEL_LOSSES = ["cross_entropy"]
RESET_EACH_BCD_VALUES = [False]
RESET_TO_SOURCE_PRETRAINED_VALUES = [False]  # official JDOT uses false
USE_ENTROPY_REG_VALUES = [True]
ENTROPY_REGS = [0.05]
USE_UNBALANCED_VALUES = [False]  # only takes effect with use_entropy_reg=False
UNBALANCED_REGS = [1.0]
SEMI_RELAXED = [False]
UNBALANCED_REG_TYPES = ["kl"]
USE_LABEL_REG_VALUES = [False]
LABEL_REGS = [1e-1]

CLASSIFIERS = ["gin"]


GRID = {
    "data_dir": DATA_DIRS,
    "derivative": DERIVATIVES,
    "edge_top_percent": EDGE_TOP_PERCENTS,
    "node_feature_mode": NODE_FEATURE_MODES,
    "source_domain": SOURCE_DOMAINS,
    "target_domain": TARGET_DOMAINS,
    "sample_mode": SAMPLE_MODES,
    "max_graphs_per_class": MAX_GRAPHS_PER_CLASS,
    "sample_seed": SAMPLE_SEEDS,
    "matrix_folder": MATRIX_FOLDERS,
    "cost_type": COST_TYPES,
    "fgw_alpha": FGW_ALPHAS,
    "cost_clip_percentile": COST_CLIP_PERCENTILES,
    "graph_cost_normalization": GRAPH_COST_NORMALIZATIONS,
    "normalize_ot_cost": NORMALIZE_OT_COST_VALUES,
    "use_symmetric_cost_cache": USE_SYMMETRIC_COST_CACHE_VALUES,
    "debug_oracle_label_loss": DEBUG_ORACLE_LABEL_LOSS,
    "seed": SEEDS,
    "in_dim": IN_DIMS,
    "hidden_dim": HIDDEN_DIMS,
    "conv_type": CONV_TYPES,
    "conv_layers": CONV_LAYERS,
    "mlp_layers": MLP_LAYERS,
    "dropout": DROPOUTS,
    "batch_size": BATCH_SIZES,
    "baseline_epochs": BASELINE_EPOCHS,
    "jdot_fit_epochs": JDOT_FIT_EPOCHS,
    "lr": LRS,
    "weight_decay": WEIGHT_DECAYS,
    "baseline_use_early_stopping": BASELINE_USE_EARLY_STOPPING_VALUES,
    "early_stopping_patience": EARLY_STOPPING_PATIENCES,
    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTAS,
    "early_stopping_val_ratio": EARLY_STOPPING_VAL_RATIOS,
    "bcd_use_early_stopping": BCD_USE_EARLY_STOPPING_VALUES,
    "bcd_early_stopping_val_ratio": BCD_EARLY_STOPPING_VAL_RATIOS,
    "bcd_early_stopping_patience": BCD_EARLY_STOPPING_PATIENCES,
    "bcd_early_stopping_min_delta": BCD_EARLY_STOPPING_MIN_DELTAS,
    "bcd_iters": BCD_ITERS,
    "log_inside_bcd": LOG_INSIDE_BCD,
    "jdot_alpha": JDOT_ALPHA,
    "label_loss": LABEL_LOSSES,
    "reset_each_bcd": RESET_EACH_BCD_VALUES,
    "reset_to_source_pretrained": RESET_TO_SOURCE_PRETRAINED_VALUES,
    "use_entropy_reg": USE_ENTROPY_REG_VALUES,
    "entropy_reg": ENTROPY_REGS,
    "use_unbalanced": USE_UNBALANCED_VALUES,
    "unbalanced_reg": UNBALANCED_REGS,
    "semi-relaxed": SEMI_RELAXED,
    "unbalanced_reg_type": UNBALANCED_REG_TYPES,
    "use_label_reg": USE_LABEL_REG_VALUES,
    "label_reg": LABEL_REGS,
    "classifier": CLASSIFIERS,
}


def is_redundant_cost_combo(params, grid):
    """FGW-only parameters do not change the Frobenius cost; run it once."""
    return params["cost_type"] == "frobenius" and params["fgw_alpha"] != grid["fgw_alpha"][0]


def graph_dataset_stats(graphs):
    labels = [int(graph.y.item()) for graph in graphs]
    label_counts = Counter(labels)
    nodes_per_graph = np.asarray([int(graph.num_nodes) for graph in graphs], dtype=np.float64)
    edge_rows_per_graph = np.asarray([int(graph.edge_index.size(1)) for graph in graphs], dtype=np.float64)

    return {
        "num_graphs": len(graphs),
        "num_nodes": int(nodes_per_graph.sum()),
        "num_edge_rows": int(edge_rows_per_graph.sum()),
        "class_0_count": int(label_counts.get(0, 0)),
        "class_1_count": int(label_counts.get(1, 0)),
        "class_1_rate": float(label_counts.get(1, 0) / max(len(labels), 1)),
        "node_feature_dim": int(graphs[0].x.size(1)),
        "nodes_per_graph_mean": float(nodes_per_graph.mean()),
        "edge_rows_per_graph_min": float(edge_rows_per_graph.min()),
        "edge_rows_per_graph_max": float(edge_rows_per_graph.max()),
        "edge_rows_per_graph_mean": float(edge_rows_per_graph.mean()),
    }


@lru_cache(maxsize=None)
def load_domain_graphs(data_dir, domain, derivative, edge_top_percent, node_feature_mode):
    graphs = load_abide_graphs(
        data_dir,
        SITE_GROUPS[domain],
        derivative=derivative,
        edge_top_percent=edge_top_percent,
        node_feature_mode=node_feature_mode,
    )
    if not graphs:
        raise ValueError(f"No ABIDE subjects found for domain {domain} (sites {SITE_GROUPS[domain]}).")
    return tuple(graphs)


def load_domains(params):
    def load(domain):
        return list(
            load_domain_graphs(
                params["data_dir"],
                domain,
                params["derivative"],
                params["edge_top_percent"],
                params["node_feature_mode"],
            )
        )

    source_all = load(params["source_domain"])
    target_all = load(params["target_domain"])
    stats_by_name = {
        params["source_domain"]: graph_dataset_stats(source_all),
        params["target_domain"]: graph_dataset_stats(target_all),
    }
    source_graphs = select_domain_graphs(source_all, params)
    target_graphs = select_domain_graphs(target_all, params)
    return stats_by_name, source_graphs, target_graphs


def domain_cache_name(domain):
    return f"{domain}({'+'.join(SITE_GROUPS[domain])})"


def make_cost_path(params):
    cost_name = params["cost_type"]
    if params["cost_type"] == "fgw":
        cost_name += (
            f"__alpha-{safe_filename_value(params['fgw_alpha'])}"
            f"__features-{safe_filename_value(params['node_feature_mode'])}"
        )
    name = (
        f"{params['derivative']}"
        f"__source-{domain_cache_name(params['source_domain'])}"
        f"__target-{domain_cache_name(params['target_domain'])}"
        f"__sample-{safe_filename_value(params['sample_mode'])}"
        f"__maxperclass-{safe_filename_value(params['max_graphs_per_class'])}"
        f"__sampleseed-{safe_filename_value(params['sample_seed'])}"
        f"__cost-{cost_name}.npz"
    )
    return PROJECT_ROOT / "cost_matrices" / params["matrix_folder"] / name


def check_cost_matrix_shape(cost_mtx, source_graphs, target_graphs, cost_path):
    expected = (len(source_graphs), len(target_graphs))
    if cost_mtx.shape != expected:
        raise ValueError(
            f"Cached cost matrix {cost_path} has shape {cost_mtx.shape}, expected {expected}; "
            "delete it to recompute."
        )
    return cost_mtx


def load_or_compute_pipeline_cost_matrix(params, source_graphs, target_graphs):
    cost_path = make_cost_path(params)
    if cost_path.exists():
        print(f"Loading cached scalar cost matrix from {cost_path}")
        cost_mtx = np.load(cost_path)["cost_mtx"]
        return check_cost_matrix_shape(cost_mtx, source_graphs, target_graphs, cost_path), cost_path, False

    if params["use_symmetric_cost_cache"]:
        swapped_path = make_cost_path(swapped_domain_params(params))
        if swapped_path.exists():
            print(f"Loading cached scalar cost matrix from {swapped_path}")
            cost_mtx = np.load(swapped_path)["cost_mtx"].T
            return check_cost_matrix_shape(cost_mtx, source_graphs, target_graphs, swapped_path), swapped_path, True

    cost_mtx = load_or_compute_cost_matrix(
        cost_path=cost_path,
        source_graphs=source_graphs,
        target_graphs=target_graphs,
        cost_type=params["cost_type"],
        fgw_alpha=params["fgw_alpha"],
        n_jobs=COST_N_JOBS,
    )
    return cost_mtx, cost_path, False


def make_run_name(params):
    cost = params["cost_type"]
    if cost == "fgw":
        cost += f"({params['fgw_alpha']})"
    return (
        f"{params['source_domain']}->{params['target_domain']} "
        f"cost={cost} "
        f"conv={params['conv_type']} "
        f"loss={resolve_classifier_label_loss(params)} "
        f"alpha={params['jdot_alpha']} "
        f"sample={params['sample_mode']} "
        f"nclass={params['max_graphs_per_class']} "
        f"clip={params['cost_clip_percentile']}"
    )


def log_run_params(params, cost_path, actual_in_dim, resolved_jdot_alpha, cost_loaded_from_transpose):
    for key, value in params.items():
        mlflow.log_param(key, str(value))
    for key, value in classifier_label_loss_metadata(params).items():
        mlflow.log_param(key, value)
    mlflow.log_param("source_sites", SITE_GROUPS[params["source_domain"]])
    mlflow.log_param("target_sites", SITE_GROUPS[params["target_domain"]])
    mlflow.log_param("cost_cache_path", str(cost_path))
    mlflow.log_param("requested_cost_cache_path", str(make_cost_path(params)))
    mlflow.log_param("cost_loaded_from_transpose", cost_loaded_from_transpose)
    mlflow.log_param("actual_in_dim", actual_in_dim)
    mlflow.log_param("resolved_jdot_alpha", resolved_jdot_alpha)


def run_experiment(params):
    set_seed(params["seed"])
    stats_by_name, source_graphs, target_graphs = load_domains(params)
    actual_in_dim = params["in_dim"] or int(source_graphs[0].x.size(1))

    if params["debug_oracle_label_loss"]:
        cost_mtx_raw = build_oracle_label_loss_matrix(source_graphs, target_graphs)
        cost_path = "debug_oracle_label_loss_no_cache"
        cost_loaded_from_transpose = False
    else:
        cost_mtx_raw, cost_path, cost_loaded_from_transpose = load_or_compute_pipeline_cost_matrix(
            params,
            source_graphs,
            target_graphs,
        )

    cost_mtx = np.asarray(cost_mtx_raw, dtype=np.float64)
    clip_percentile = params["cost_clip_percentile"]
    if clip_percentile is not None:
        clip_value = float(np.percentile(finite_values(cost_mtx), clip_percentile))
        cost_mtx_used = np.minimum(cost_mtx, clip_value)
    else:
        clip_value = None
        cost_mtx_used = cost_mtx.copy()
    cost_mtx_used = normalize_graph_cost(cost_mtx_used, params["graph_cost_normalization"])

    jdot_alpha = compute_jdot_alpha(params["jdot_alpha"], cost_mtx_used)
    log_run_params(params, cost_path, actual_in_dim, jdot_alpha, cost_loaded_from_transpose)
    mlflow.log_param("cost_clip_value", clip_value)

    log_raw_dataset_stats(f"raw_{params['source_domain']}", stats_by_name[params["source_domain"]])
    log_raw_dataset_stats(f"raw_{params['target_domain']}", stats_by_name[params["target_domain"]])
    log_class_label_stats("source", source_graphs, CLASS_LABELS)
    log_class_label_stats("target", target_graphs, CLASS_LABELS)
    log_stats("raw_cost_mtx", cost_mtx)
    log_stats("used_cost_mtx", cost_mtx_used)
    log_histogram(cost_mtx, "Raw Graph Cost Values", "plots/raw_cost_histogram.png")
    log_histogram(cost_mtx_used, "Used Graph Cost Values", "plots/used_cost_histogram.png")
    log_matrix_image(cost_mtx, "Raw Graph Cost Matrix", "plots/raw_cost_matrix.png")
    log_matrix_image(cost_mtx_used, "Used Graph Cost Matrix", "plots/used_cost_matrix.png")

    src_loader = DataLoader(source_graphs, batch_size=params["batch_size"], shuffle=False)
    tgt_loader = DataLoader(target_graphs, batch_size=params["batch_size"], shuffle=False)

    ys = np.array([int(g.y.item()) for g in source_graphs], dtype=np.float32).reshape(-1, 1)
    w_s = ot.unif(len(source_graphs))
    w_t = ot.unif(len(target_graphs))

    classifier = make_jdot_classifier(
        params,
        source_graphs=source_graphs,
        target_graphs=target_graphs,
        src_loader=src_loader,
        tgt_loader=tgt_loader,
        num_classes=2,
        in_dim=actual_in_dim,
    )
    baseline_metrics = classifier.baseline_metrics()
    log_prefixed_metrics("baseline", baseline_metrics)
    log_confusion_matrix(
        {
            "confusion_matrix": baseline_metrics["target_confusion_matrix"],
        },
        "Baseline Target Confusion Matrix",
        "plots/baseline_target_confusion_matrix.png",
        CLASS_LABELS,
    )

    y_pred = classifier.predict_probs()
    label_loss_matrix = classifier.compute_ot_label_loss_matrix(ys, predicted_probs=y_pred)
    jdot_cost_matrix = jdot_alpha * cost_mtx_used + label_loss_matrix
    log_stats("initial_label_loss", label_loss_matrix)
    log_stats("initial_jdot_cost_matrix", jdot_cost_matrix)
    log_matrix_image(jdot_cost_matrix, "Initial JDOT Cost Matrix", "plots/initial_jdot_cost_matrix.png")
    log_matrix_image(label_loss_matrix, "Initial Label Loss Matrix", "plots/initial_label_loss_matrix.png")

    gamma = None
    y_soft = None
    bcd_target_accuracies = []
    bcd_target_f1_scores = []
    if params["log_inside_bcd"]:
        bcd_metrics = classifier.evaluate_target()
        bcd_target_accuracy = bcd_metrics["accuracy"]
        bcd_target_f1 = bcd_metrics["f1"]
        bcd_target_accuracies.append(float(bcd_target_accuracy))
        bcd_target_f1_scores.append(float(bcd_target_f1))
        mlflow.log_metric("bcd_target_accuracy", float(bcd_target_accuracy), step=0)
        mlflow.log_metric("bcd_target_f1", float(bcd_target_f1), step=0)
    for bcd_step in range(1, params["bcd_iters"] + 1):
        if params["normalize_ot_cost"]:
            ot_cost, ot_cost_scale = normalize_ot_cost(jdot_cost_matrix)
        else:
            ot_cost, ot_cost_scale = jdot_cost_matrix, 1.0
        mlflow.log_metric("ot_cost_normalization_scale", float(ot_cost_scale), step=bcd_step)
        log_stats("ot_cost", ot_cost, step=bcd_step)
        gamma = solve_ot(params, w_s, w_t, ot_cost, ys=ys, numItermax=1000000)
        if gamma is None:
            mlflow.set_tag("skipped", "true")
            mlflow.set_tag("skip_reason", "ot_warning")
            mlflow.log_metric("ot_warning_skip", 1, step=bcd_step)
            mlflow.log_metric("ot_warning_bcd_step", bcd_step)
            print(
                f"Skipping {params['source_domain']}->{params['target_domain']} "
                f"at BCD step {bcd_step} due to OT warning."
            )
            return

        if params["use_unbalanced"]:
            log_ot_marginal_violations(gamma, w_s, w_t, step=bcd_step)

        y_soft, jdot_training_info = classifier.fit_jdot(gamma, ys, bcd_step)
        if params["bcd_use_early_stopping"]:
            mlflow.log_metric("bcd_refresh_epochs_run", jdot_training_info["epochs_run"], step=bcd_step)
            mlflow.log_metric("bcd_refresh_best_epoch", jdot_training_info["best_epoch"], step=bcd_step)
            mlflow.log_metric("bcd_refresh_best_val_loss", jdot_training_info["best_val_loss"], step=bcd_step)
            mlflow.log_metric(
                "bcd_refresh_early_stopping_triggered",
                float(jdot_training_info["early_stopping_triggered"]),
                step=bcd_step,
            )

        y_pred = classifier.predict_probs()
        label_loss_matrix = classifier.compute_ot_label_loss_matrix(ys, predicted_probs=y_pred)
        jdot_cost_matrix = jdot_alpha * cost_mtx_used + label_loss_matrix
        for metric_name, metric_value in compute_jdot_terms(gamma, cost_mtx_used, label_loss_matrix, jdot_alpha).items():
            mlflow.log_metric(metric_name, metric_value, step=bcd_step)
        log_predicted_class_distribution(y_pred, num_classes=2, step=bcd_step)

        if params["log_inside_bcd"]:
            bcd_metrics = classifier.evaluate_target()
            bcd_target_accuracy = bcd_metrics["accuracy"]
            bcd_target_f1 = bcd_metrics["f1"]
            bcd_target_accuracies.append(float(bcd_target_accuracy))
            bcd_target_f1_scores.append(float(bcd_target_f1))
            mlflow.log_metric("bcd_target_accuracy", float(bcd_target_accuracy), step=bcd_step)
            mlflow.log_metric("bcd_target_f1", float(bcd_target_f1), step=bcd_step)

        mlflow.log_metric("pseudo_label_mean", float(np.mean(y_soft)), step=bcd_step)
        mlflow.log_metric("pseudo_label_std", float(np.std(y_soft)), step=bcd_step)
        mlflow.log_metric("label_loss_mean", float(np.mean(label_loss_matrix)), step=bcd_step)
        mlflow.log_metric("jdot_cost_matrix_mean", float(np.mean(jdot_cost_matrix)), step=bcd_step)

    if params["log_inside_bcd"]:
        mlflow.log_dict(
            {
                "bcd_iteration": list(range(0, params["bcd_iters"] + 1)),
                "target_accuracy": bcd_target_accuracies,
                "target_f1": bcd_target_f1_scores,
            },
            "metrics/bcd_target_metrics.json",
        )

    target_metrics = classifier.evaluate_target()
    source_metrics = classifier.evaluate_source()
    log_prefixed_metrics("final_target", target_metrics)
    log_prefixed_metrics("final_source", source_metrics)
    log_confusion_matrix(
        target_metrics,
        "Final Target Confusion Matrix",
        "plots/final_target_confusion_matrix.png",
        CLASS_LABELS,
    )
    log_confusion_matrix(
        source_metrics,
        "Final Source Confusion Matrix",
        "plots/final_source_confusion_matrix.png",
        CLASS_LABELS,
    )
    if gamma is not None:
        log_matrix_image(gamma, "Final Transport Plan", "plots/final_transport_plan.png")
    log_matrix_image(jdot_cost_matrix, "Final JDOT Cost Matrix", "plots/final_jdot_cost_matrix.png")
    log_matrix_image(label_loss_matrix, "Final Label Loss Matrix", "plots/final_label_loss_matrix.png")

    mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source_code")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    for params in iter_grid(GRID):
        if is_redundant_cost_combo(params, GRID):
            continue
        with mlflow.start_run(run_name=make_run_name(params)):
            run_experiment(params)


if __name__ == "__main__":
    main()
