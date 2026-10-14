import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch_geometric.loader import DataLoader

from GJDOT.gjdot_gnn import GNNClassifier
from GJDOT.gjdot_ot import label_loss_per_sample

def build_gnn(
    in_dim,
    num_classes,
    hidden_dim=64,
    dropout=0.0,
    num_layers=3,
    mlp_layers=2,
    conv_type="gin",
):
    return GNNClassifier(
        in_channels=in_dim,
        hidden=hidden_dim,
        conv_type=conv_type,
        dropout=dropout,
        use_batchnorm=True,
        mlp_hidden=hidden_dim,
        mlp_layers=mlp_layers,
        num_layers=num_layers,
        num_classes=num_classes,
    )


def clone_state_dict(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}


def classification_metrics(y_true, y_pred, num_classes=2, loss=0.0):
    """Compute the metric dict from hard predictions (used by :func:`evaluate_classifier`)."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0

    if num_classes == 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(value) for value in cm.ravel()]
        class_recalls = []
        if tp + fn > 0:
            class_recalls.append(tp / (tp + fn))
        if tn + fp > 0:
            class_recalls.append(tn / (tn + fp))
        balanced_acc = float(np.mean(class_recalls)) if class_recalls else 0.0
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        _, _, f1_macro, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
        return {
            "loss": float(loss),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "f1": float(f1_macro),
            "f1_macro": float(f1_macro),
            "confusion_matrix": cm.tolist(),
        }

    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    return {
        "loss": float(loss),
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "f1_macro": float(f1_macro),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
        ).tolist(),
    }


def evaluate_classifier(model, loader, num_classes=2, label_loss="cross_entropy"):
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            if num_classes == 2:
                y = batch.y.view(-1).float()
                probs = torch.sigmoid(logits)
                total_loss += float(_classifier_loss(logits, y, num_classes, label_loss, reduction="sum").item())
                pred = (probs >= 0.5).long()
                y_true.extend(y.long().cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())
            else:
                y = batch.y.view(-1).long()
                total_loss += float(_classifier_loss(logits, y, num_classes, label_loss, reduction="sum").item())
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    loss = total_loss / max(len(y_true), 1)
    return classification_metrics(y_true, y_pred, num_classes=num_classes, loss=loss)


def predict_probs(model, loader, num_classes=2):
    """Predict graph-level probabilities in loader order.

    Binary classifiers return positive-class probabilities with shape (n, 1);
    multiclass classifiers return softmax probabilities with shape (n, C).
    """
    device = next(model.parameters()).device
    model.eval()

    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            if num_classes == 2:
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            else:
                probs.append(F.softmax(logits, dim=1).cpu().numpy())

    if num_classes == 2:
        return np.asarray(probs, dtype=np.float32).reshape(-1, 1)
    else:
        return np.vstack(probs).astype(np.float32)


def predict_scores(model, loader, num_classes=2):
    """Predict graph-level raw decision scores in loader order."""
    device = next(model.parameters()).device
    model.eval()

    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            if num_classes == 2:
                scores.append(logits.cpu().numpy().reshape(-1, 1))
            else:
                scores.append(logits.cpu().numpy())

    return np.vstack(scores).astype(np.float32)


def train_jdot_classifier(
    model,
    opt,
    tgt_loader,
    gamma,
    y_source,
    num_classes=2,
    epochs=1,
    log=False,
    label_loss="cross_entropy",
    use_early_stopping=False,
    val_ratio=0.1,
    patience=5,
    min_delta=0.0,
    val_split_seed=None,
    return_training_info=False,
):
    device = next(model.parameters()).device
    gamma_tensor = torch.as_tensor(gamma, dtype=torch.float32, device=device)
    y_source_tensor = torch.as_tensor(y_source, dtype=torch.float32, device=device)

    col_mass = gamma_tensor.sum(dim=0).clamp_min(1e-12)
    if num_classes == 2:
        y_source_tensor = y_source_tensor.reshape(-1)
        y_soft = (gamma_tensor.t() @ y_source_tensor) / col_mass
        y_soft = y_soft.clamp_(0.0, 1.0)
    else:
        if y_source_tensor.ndim == 1 or y_source_tensor.shape[-1] != num_classes:
            y_source_tensor = F.one_hot(y_source_tensor.long().view(-1), num_classes=num_classes).float()
        y_soft = (gamma_tensor.t() @ y_source_tensor) / col_mass[:, None]
        y_soft = y_soft.clamp_min(0.0)
        y_soft = y_soft / y_soft.sum(dim=1, keepdim=True).clamp_min(1e-12)

    dataset = tgt_loader.dataset
    batch_size = tgt_loader.batch_size
    m = len(dataset)

    weighted_graphs = []
    for j in range(m):
        graph = dataset[j].clone()
        if num_classes == 2:
            graph.pseudo_y = torch.tensor([float(y_soft[j].item())], dtype=torch.float32)
        else:
            graph.pseudo_y = y_soft[j].detach().cpu().reshape(1, num_classes)
        graph.ot_w = torch.tensor([float(col_mass[j].item())], dtype=torch.float32)
        weighted_graphs.append(graph)

    if use_early_stopping and m > 1:
        n_val = max(1, int(round(val_ratio * m)))
        if n_val >= m:
            n_val = m - 1
        generator = None
        if val_split_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(val_split_seed))
        perm = torch.randperm(m, generator=generator)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        train_graphs = [weighted_graphs[int(i)] for i in train_idx]
        val_graphs = [weighted_graphs[int(i)] for i in val_idx]
    else:
        train_graphs = weighted_graphs
        val_graphs = weighted_graphs

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    last_eval_loss = float("inf")
    best_eval_loss = float("inf")
    best_epoch = 0
    epochs_run = 0
    no_improve = 0
    early_stopping_triggered = False
    best_state = None

    for ep in range(1, epochs + 1):
        epochs_run = ep
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            if num_classes == 2:
                y_b = batch.pseudo_y.view(-1)
                per_graph_label_loss = label_loss_per_sample(
                    _output_for_label_loss(logits, num_classes, label_loss),
                    y_b,
                    label_loss=label_loss,
                )
            else:
                y_b = batch.pseudo_y.view(-1, num_classes)
                per_graph_label_loss = label_loss_per_sample(
                    _output_for_label_loss(logits, num_classes, label_loss),
                    y_b,
                    label_loss=label_loss,
                )
            w_b = batch.ot_w.view(-1)
            loss = (w_b * per_graph_label_loss).sum() / w_b.sum().clamp_min(1e-12)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())

        model.eval()
        eval_sum = 0.0
        eval_mass = 0.0
        
        # For analysis purposes, evaluate the current iteration on the dataset.
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device)
                logits = model(batch.x.float(), batch.edge_index, batch.batch)
                if num_classes == 2:
                    y_b = batch.pseudo_y.view(-1)
                    per_graph_label_loss = label_loss_per_sample(
                        _output_for_label_loss(logits, num_classes, label_loss),
                        y_b,
                        label_loss=label_loss,
                    )
                else:
                    y_b = batch.pseudo_y.view(-1, num_classes)
                    per_graph_label_loss = label_loss_per_sample(
                        _output_for_label_loss(logits, num_classes, label_loss),
                        y_b,
                        label_loss=label_loss,
                    )
                w_b = batch.ot_w.view(-1)
                eval_sum += float((w_b * per_graph_label_loss).sum().item())
                eval_mass += float(w_b.sum().item())
        last_eval_loss = eval_sum / max(eval_mass, 1e-12)

        if log and (ep == 1 or ep % 5 == 0):
            print(f"JDOT step {ep:03d} | train_loss {epoch_loss:.4f} eval_loss {last_eval_loss:.4f}")

        if use_early_stopping:
            if last_eval_loss < (best_eval_loss - min_delta):
                best_eval_loss = last_eval_loss
                best_epoch = ep
                no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    early_stopping_triggered = True
                    break

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    if log:
        if use_early_stopping:
            print(
                f"JDOT finished {epochs_run:03d} epoch(s); "
                f"best epoch {best_epoch:03d} with val_loss {best_eval_loss:.4f}"
            )
        else:
            print(f"JDOT finished {epochs:03d} epoch(s); final eval_loss {last_eval_loss:.4f}")

    y_soft_np = y_soft.detach().cpu().numpy()
    if num_classes == 2:
        y_soft_np = y_soft_np.reshape(-1, 1)
    if return_training_info:
        training_info = {
            "use_early_stopping": bool(use_early_stopping),
            "epochs_run": int(epochs_run),
            "best_epoch": int(best_epoch if use_early_stopping else epochs_run),
            "best_val_loss": float(best_eval_loss if use_early_stopping else last_eval_loss),
            "early_stopping_triggered": bool(early_stopping_triggered),
            "train_size": int(len(train_graphs)),
            "val_size": int(len(val_graphs)),
        }
        return model, y_soft_np, training_info
    return model, y_soft_np


def _output_for_label_loss(logits, num_classes, label_loss):
    if label_loss == "squared_hinge":
        return logits
    if label_loss == "cross_entropy":
        if num_classes == 2:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)
    raise ValueError(f"Unknown label_loss: {label_loss}")


def _classifier_targets(y, num_classes):
    if num_classes == 2:
        return y.view(-1).float()
    return F.one_hot(y.view(-1).long(), num_classes=num_classes).float()


def _classifier_loss(logits, y, num_classes, label_loss="cross_entropy", reduction="mean"):
    if label_loss == "cross_entropy":
        if num_classes == 2:
            return F.binary_cross_entropy(torch.sigmoid(logits), y.view(-1).float(), reduction=reduction)
        return F.cross_entropy(logits, y.view(-1).long(), reduction=reduction)

    targets = _classifier_targets(y, num_classes)
    per_sample_loss = label_loss_per_sample(logits, targets, label_loss=label_loss)
    if reduction == "sum":
        return per_sample_loss.sum()
    if reduction == "mean":
        return per_sample_loss.mean()
    if reduction == "none":
        return per_sample_loss
    raise ValueError(f"Unknown reduction: {reduction}")


def run_source_only_baseline(
    source_graphs,
    target_graphs,
    in_dim,
    num_classes,
    hidden_dim=64,
    conv_layers=3,
    mlp_layers=2,
    conv_type="gin",
    dropout=0.0,
    batch_size=64,
    epochs=100,
    lr=1e-4,
    weight_decay=0.0,
    use_early_stopping=False,
    val_ratio=0.1,
    patience=5,
    min_delta=0.0,
    label_loss="cross_entropy",
):
    model = build_gnn(
        in_dim=in_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=conv_layers,
        mlp_layers=mlp_layers,
        conv_type=conv_type,
    )
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_src = len(source_graphs)
    if use_early_stopping:
        n_val = max(1, int(round(val_ratio * n_src)))
        if n_val >= n_src:
            n_val = n_src - 1
        perm = torch.randperm(n_src)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        train_graphs = [source_graphs[int(i)] for i in train_idx]
        val_graphs = [source_graphs[int(i)] for i in val_idx]
    else:
        train_graphs = source_graphs
        val_graphs = source_graphs

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    src_loader = DataLoader(source_graphs, batch_size=batch_size, shuffle=False)
    tgt_loader = DataLoader(target_graphs, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            y = batch.y.view(-1)
            loss = _classifier_loss(logits, y, num_classes, label_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x.float(), batch.edge_index, batch.batch)
                y = batch.y.view(-1)
                val_loss_sum += float(
                    _classifier_loss(logits, y, num_classes, label_loss, reduction="sum").item()
                )
                val_count += y.numel()
        val_loss = val_loss_sum / max(val_count, 1)

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = ep
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif use_early_stopping:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    source_metrics = evaluate_classifier(model, src_loader, num_classes=num_classes, label_loss=label_loss)
    target_metrics = evaluate_classifier(model, tgt_loader, num_classes=num_classes, label_loss=label_loss)

    metrics = {}
    metrics.update({f"source_{key}": value for key, value in source_metrics.items()})
    metrics.update({f"target_{key}": value for key, value in target_metrics.items()})
    metrics["best_val_loss"] = best_val_loss
    metrics["best_epoch"] = best_epoch
    return model, metrics
