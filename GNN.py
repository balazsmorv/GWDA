import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GINConv, GATConv,
    BatchNorm, global_mean_pool
)

class GNN3LayerBinary(nn.Module):
    """
    Three-layer GNN (+ MLP head) for binary graph classification.
    - Supports: 'gcn', 'sage', 'gin', 'gat'
    - Returns a single logit per graph.
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
        gat_heads: int = 4,        # only used if conv_type='gat'
    ):
        super().__init__()
        self.dropout = dropout
        self.use_bn = use_batchnorm
        self.conv_type = conv_type.lower()

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

        # 3 conv layers
        self.conv1 = make_conv(in_channels, hidden)
        self.conv2 = make_conv(hidden, hidden)
        self.conv3 = make_conv(hidden, hidden)

        # optional batch norms
        if self.use_bn:
            self.bn1 = BatchNorm(hidden)
            self.bn2 = BatchNorm(hidden)
            self.bn3 = BatchNorm(hidden)

        # MLP head: hidden -> ... -> 1 logit
        mlp = []
        cin = hidden
        for _ in range(max(mlp_layers - 1, 0)):
            mlp += [nn.Linear(cin, mlp_hidden), nn.ReLU(), nn.Dropout(dropout)]
            cin = mlp_hidden
        mlp += [nn.Linear(cin, 1)]
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
        # 3 conv blocks
        x = self._block(x, self.conv1, getattr(self, "bn1", None), edge_index, edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self._block(x, self.conv2, getattr(self, "bn2", None), edge_index, edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self._block(x, self.conv3, getattr(self, "bn3", None), edge_index, edge_weight)

        # graph-level pooling
        x = global_mean_pool(x, batch)

        # classification head -> single logit
        logit = self.mlp(x).squeeze(-1)  # (batch_size,)
        return logit