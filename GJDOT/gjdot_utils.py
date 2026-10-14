import itertools
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_filename_value(value):
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8g}".replace(".", "_").replace("-", "m")
    return str(value).replace(".", "_").replace("/", "-").replace(" ", "")


def iter_grid(grid):
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        if params["source_domain"] == params["target_domain"]:
            continue
        yield params


def stratified_sample_graphs(graphs, max_graphs_per_class, seed):
    if max_graphs_per_class is None:
        return list(graphs)

    rng = np.random.default_rng(seed)
    selected = []
    labels = [int(graph.y.item()) for graph in graphs]
    for label in sorted(set(labels)):
        label_indices = np.asarray([idx for idx, y in enumerate(labels) if y == label], dtype=np.int64)
        rng.shuffle(label_indices)
        selected.extend(label_indices[: min(max_graphs_per_class, len(label_indices))].tolist())

    return [graphs[idx] for idx in sorted(selected)]


def select_domain_graphs(graphs, params):
    if params["sample_mode"] == "full":
        return list(graphs)
    if params["sample_mode"] == "stratified":
        return stratified_sample_graphs(
            graphs,
            params["max_graphs_per_class"],
            params["sample_seed"],
        )
    raise ValueError(f"Unknown sample_mode: {params['sample_mode']}")
