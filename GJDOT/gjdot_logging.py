import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def finite_values(values):
    arr = np.asarray(values).ravel()
    return arr[np.isfinite(arr)]


def numeric_stats(values):
    vals = finite_values(values)
    if vals.size == 0:
        return {}
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
    }


def log_prefixed_metrics(prefix, metrics, step=None):
    serializable = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        serializable[key] = value
        if isinstance(value, (int, float, bool)):
            mlflow.log_metric(f"{prefix}_{key}", float(value), step=step)
    mlflow.log_dict(serializable, f"metrics/{prefix}.json")


def log_stats(prefix, values, step=None):
    for key, value in numeric_stats(values).items():
        mlflow.log_metric(f"{prefix}_{key}", value, step=step)


def ot_marginal_violation_metrics(gamma, w_s, w_t):
    gamma = np.asarray(gamma, dtype=np.float64)
    w_s = np.asarray(w_s, dtype=np.float64).reshape(-1)
    w_t = np.asarray(w_t, dtype=np.float64).reshape(-1)

    source_diff = gamma.sum(axis=1) - w_s
    target_diff = gamma.sum(axis=0) - w_t
    total_mass = float(np.sum(gamma))
    expected_total_mass = 0.5 * float(np.sum(w_s) + np.sum(w_t))

    return {
        "ot_total_mass": total_mass,
        "ot_total_mass_delta": total_mass - expected_total_mass,
        "ot_source_marginal_violation_l1": float(np.sum(np.abs(source_diff))),
        "ot_source_marginal_violation_l2": float(np.linalg.norm(source_diff, ord=2)),
        "ot_source_marginal_violation_linf": float(np.max(np.abs(source_diff))) if source_diff.size else 0.0,
        "ot_target_marginal_violation_l1": float(np.sum(np.abs(target_diff))),
        "ot_target_marginal_violation_l2": float(np.linalg.norm(target_diff, ord=2)),
        "ot_target_marginal_violation_linf": float(np.max(np.abs(target_diff))) if target_diff.size else 0.0,
    }


def log_ot_marginal_violations(gamma, w_s, w_t, step=None):
    for key, value in ot_marginal_violation_metrics(gamma, w_s, w_t).items():
        mlflow.log_metric(key, value, step=step)


def log_histogram(values, title, artifact_file, bins=100):
    vals = finite_values(values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    mlflow.log_figure(fig, artifact_file)
    plt.close(fig)


def log_matrix_image(matrix, title, artifact_file):
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    mlflow.log_figure(fig, artifact_file)
    plt.close(fig)


def log_confusion_matrix(metrics, title, artifact_file, class_labels, figsize=(6, 5), xticks_rotation=None):
    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=figsize)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels,
    )
    kwargs = {"cmap": "Blues", "ax": ax, "values_format": "d"}
    if xticks_rotation is not None:
        kwargs["xticks_rotation"] = xticks_rotation
    display.plot(**kwargs)
    ax.set_title(title)
    fig.tight_layout()
    mlflow.log_figure(fig, artifact_file)
    plt.close(fig)


def log_raw_dataset_stats(prefix, stats):
    for key, value in stats.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            mlflow.log_metric(f"{prefix}_{key}", float(value))


def log_predicted_class_distribution(y_pred, num_classes, step):
    y_pred = np.asarray(y_pred)
    if num_classes == 2:
        labels = (y_pred.reshape(-1) >= 0.5).astype(np.int64)
    else:
        labels = np.argmax(y_pred, axis=1).astype(np.int64)

    total = max(int(labels.size), 1)
    for class_idx in range(num_classes):
        count = int(np.sum(labels == class_idx))
        mlflow.log_metric(f"target_predicted_class_{class_idx}_count", count, step=step)
        mlflow.log_metric(f"target_predicted_class_{class_idx}_rate", count / total, step=step)


def log_class_label_stats(prefix, graphs, class_to_index):
    labels = [int(graph.y.item()) for graph in graphs]
    mlflow.log_metric(f"{prefix}_num_graphs", len(graphs))

    if isinstance(class_to_index, (list, tuple)):
        class_to_index = {label: idx for idx, label in enumerate(class_to_index)}

    for label, idx in class_to_index.items():
        mlflow.log_metric(f"{prefix}_class_{label}_count", int(labels.count(idx)))

    class_indices = set(class_to_index.values())
    if len(class_to_index) == 2 and class_indices == {0, 1}:
        count_0 = int(labels.count(0))
        count_1 = int(labels.count(1))
        mlflow.log_metric(f"{prefix}_label_0_count", count_0)
        mlflow.log_metric(f"{prefix}_label_1_count", count_1)
        mlflow.log_metric(f"{prefix}_label_1_rate", count_1 / max(count_0 + count_1, 1))
