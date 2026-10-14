import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GINConv, GATConv,
    BatchNorm, global_add_pool
)

class GNNClassifier(nn.Module):
    """
    Configurable N-layer GNN (+ MLP head) for graph classification.
    - Supports: 'gcn', 'sage', 'gin', 'gat'
    - num_classes=2 is binary classification and returns one logit per graph.
    - num_classes>2 is multiclass classification and returns class logits.
    """
    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        conv_type: str = "gcn",
        dropout: float = 0.5,
        use_batchnorm: bool = True,
        mlp_hidden: int = 64,
        mlp_layers: int = 2,
        num_layers: int = 3,
        num_classes: int = 2,
        gat_heads: int = 4,        # only used if conv_type='gat'
    ):
        super().__init__()
        self.dropout = dropout
        self.use_bn = use_batchnorm
        self.conv_type = conv_type.lower()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.out_channels = 1 if num_classes == 2 else num_classes

        def make_conv(cin, cout):
            if self.conv_type == "gcn":
                return GCNConv(cin, cout, normalize=True)
            if self.conv_type == "sage":
                return SAGEConv(cin, cout)
            if self.conv_type == "gin":
                mlp = nn.Sequential(nn.Linear(cin, cout), nn.ReLU(), nn.Linear(cout, cout))
                return GINConv(mlp)
            if self.conv_type == "gat":
                # multi-head; concat heads -> cout
                head_dim = cout // gat_heads
                assert head_dim * gat_heads == cout, "hidden must be divisible by gat_heads"
                return GATConv(cin, head_dim, heads=gat_heads, concat=True, add_self_loops=True)
            raise ValueError(f"Unknown conv_type: {conv_type}")

        for layer_idx in range(1, self.num_layers + 1):
            cin = in_channels if layer_idx == 1 else hidden
            setattr(self, f"conv{layer_idx}", make_conv(cin, hidden))
            if self.use_bn:
                setattr(self, f"bn{layer_idx}", BatchNorm(hidden))

        mlp = []
        cin = hidden
        for _ in range(max(mlp_layers - 1, 0)):
            mlp += [nn.Linear(cin, mlp_hidden), nn.ReLU(), nn.Dropout(dropout)]
            cin = mlp_hidden
        mlp += [nn.Linear(cin, self.out_channels)]
        self.mlp = nn.Sequential(*mlp)

    def _block(self, x, conv, bn, edge_index, edge_weight=None):
        if self.conv_type in ("gcn",):
            x = conv(x, edge_index, edge_weight)
        else:
            x = conv(x, edge_index)
        if self.use_bn:
            x = bn(x)
        return F.relu(x)

    def forward(self, x, edge_index, batch, edge_weight=None):
        for layer_idx in range(1, self.num_layers + 1):
            conv = getattr(self, f"conv{layer_idx}")
            bn = getattr(self, f"bn{layer_idx}", None)
            x = self._block(x, conv, bn, edge_index, edge_weight)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # graph-level pooling
        x = global_add_pool(x, batch)

        logits = self.mlp(x)
        if self.num_classes == 2:
            return logits.squeeze(-1)  # (batch_size,)
        return logits
