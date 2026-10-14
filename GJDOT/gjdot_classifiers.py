"""Pluggable JDOT classifiers for the GJDOT block-coordinate-descent loop.

Ported from the gjdot repository (commit 26c0075). The BCD loop alternates
between (a) solving the sample-level OT problem for the coupling ``gamma`` and
(b) retraining a classifier ``f`` on the target graphs using the OT-transported
source labels -- JDOT's fixed-coupling learning problem, Eq. (7):

    min_f  sum_ij gamma_ij L(y_i^s, f(x_j^t)) + Omega(f).

Only the GNN classifier (``classifier="gin"``) is ported: the WL subtree kernel
classifiers of the original need discrete node labels, which connectome graphs
do not have. The classifier owns the label loss used both for its training and
in the OT ground cost, so both BCD steps optimise the same JDOT objective.
"""

import torch

from GJDOT.gjdot_models import (
    build_gnn,
    clone_state_dict,
    evaluate_classifier,
    predict_probs,
    predict_scores,
    run_source_only_baseline,
    train_jdot_classifier,
)
from GJDOT.gjdot_ot import label_loss


__all__ = [
    "BaseJDOTClassifier",
    "GINJDOTClassifier",
    "classifier_label_loss_metadata",
    "make_jdot_classifier",
    "resolve_classifier_label_loss",
]


def resolve_classifier_label_loss(params):
    """Return the native label loss shared by classifier fitting and OT."""
    classifier_type = params.get("classifier", "gin")
    if classifier_type == "gin":
        return params.get("label_loss", "cross_entropy")
    raise ValueError(f"Unknown classifier={classifier_type!r}; expected 'gin'.")


def classifier_label_loss_metadata(params):
    """MLflow parameters describing the effective losses for a classifier."""
    resolved_loss = resolve_classifier_label_loss(params)
    return {
        "classifier_training_loss": resolved_loss,
        "ot_label_loss": resolved_loss,
    }


class BaseJDOTClassifier:
    """Interface used by the BCD loop. Implementations wrap a concrete model.

    ``predict_probs`` / ``predict_scores`` return target predictions in loader
    order matching the GIN contract: shape ``(N_t, 1)`` for binary (positive
    class), ``(N_t, C)`` for multiclass.
    """

    def baseline_metrics(self):
        raise NotImplementedError

    def predict_probs(self):
        raise NotImplementedError

    def predict_scores(self):
        raise NotImplementedError

    @property
    def training_label_loss_name(self):
        raise NotImplementedError

    @property
    def ot_label_loss_name(self):
        raise NotImplementedError

    def compute_ot_label_loss_matrix(self, y_source, predicted_probs=None):
        """Pairwise source-label/target-prediction loss for the OT step."""
        raise NotImplementedError

    def fit_jdot(self, gamma, y_source, bcd_step):
        """Fixed-coupling learning problem (JDOT Eq. 7). Returns (y_soft, info)."""
        raise NotImplementedError

    def evaluate_target(self):
        raise NotImplementedError

    def evaluate_source(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# GIN adapter -- thin wrapper
# ---------------------------------------------------------------------------
class GINJDOTClassifier(BaseJDOTClassifier):
    def __init__(self, params, source_graphs, target_graphs, src_loader, tgt_loader, num_classes, in_dim):
        self.params = params
        self.num_classes = int(num_classes)
        self.in_dim = in_dim
        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self._conv_layers = params.get("conv_layers", 3)

        self.model, self._baseline_metrics = run_source_only_baseline(
            source_graphs=source_graphs,
            target_graphs=target_graphs,
            in_dim=in_dim,
            num_classes=self.num_classes,
            hidden_dim=params["hidden_dim"],
            conv_layers=self._conv_layers,
            conv_type=params["conv_type"],
            mlp_layers=params["mlp_layers"],
            dropout=params["dropout"],
            batch_size=params["batch_size"],
            epochs=params["baseline_epochs"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            use_early_stopping=params["baseline_use_early_stopping"],
            val_ratio=params["early_stopping_val_ratio"],
            patience=params["early_stopping_patience"],
            min_delta=params["early_stopping_min_delta"],
            label_loss=params["label_loss"],
        )
        self._pretrained_state = clone_state_dict(self.model.state_dict())

    def baseline_metrics(self):
        return self._baseline_metrics

    def predict_probs(self):
        return predict_probs(self.model, self.tgt_loader, num_classes=self.num_classes)

    def predict_scores(self):
        return predict_scores(self.model, self.tgt_loader, num_classes=self.num_classes)

    @property
    def training_label_loss_name(self):
        return resolve_classifier_label_loss(self.params)

    @property
    def ot_label_loss_name(self):
        return self.training_label_loss_name

    def compute_ot_label_loss_matrix(self, y_source, predicted_probs=None):
        if self.ot_label_loss_name == "squared_hinge":
            predictions = self.predict_scores()
        else:
            predictions = predicted_probs if predicted_probs is not None else self.predict_probs()
        return label_loss(y_source, predictions, label_loss_type=self.ot_label_loss_name)

    def fit_jdot(self, gamma, y_source, bcd_step):
        params = self.params
        if params["reset_each_bcd"]:
            self.model = build_gnn(
                in_dim=self.in_dim,
                num_classes=self.num_classes,
                hidden_dim=params["hidden_dim"],
                dropout=params["dropout"],
                num_layers=self._conv_layers,
                mlp_layers=params["mlp_layers"],
                conv_type=params["conv_type"],
            )
            if params["reset_to_source_pretrained"]:
                self.model.load_state_dict(self._pretrained_state)

        opt = torch.optim.Adam(self.model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        self.model, y_soft, training_info = train_jdot_classifier(
            self.model,
            opt,
            self.tgt_loader,
            gamma,
            y_source,
            num_classes=self.num_classes,
            epochs=params["jdot_fit_epochs"],
            log=True,
            label_loss=params["label_loss"],
            use_early_stopping=params["bcd_use_early_stopping"],
            val_ratio=params["bcd_early_stopping_val_ratio"],
            patience=params["bcd_early_stopping_patience"],
            min_delta=params["bcd_early_stopping_min_delta"],
            val_split_seed=params["seed"] + bcd_step,
            return_training_info=True,
        )
        return y_soft, training_info

    def evaluate_target(self):
        return evaluate_classifier(
            self.model, self.tgt_loader, num_classes=self.num_classes, label_loss=self.params["label_loss"]
        )

    def evaluate_source(self):
        return evaluate_classifier(
            self.model, self.src_loader, num_classes=self.num_classes, label_loss=self.params["label_loss"]
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_jdot_classifier(
    params,
    *,
    source_graphs,
    target_graphs,
    src_loader,
    tgt_loader,
    num_classes,
    in_dim,
):
    """Build the classifier selected by ``params['classifier']`` (source-pretrained)."""
    classifier_type = params.get("classifier", "gin")

    if classifier_type == "gin":
        return GINJDOTClassifier(
            params, source_graphs, target_graphs, src_loader, tgt_loader, num_classes, in_dim
        )

    raise ValueError(f"Unknown classifier={classifier_type!r}; expected 'gin'.")
