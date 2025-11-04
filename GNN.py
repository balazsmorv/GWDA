import torch
from torch import nn
from torch_geometric.nn import aggr
from torch.nn import ModuleList
from torch.nn import CrossEntropyLoss
softmax = torch.nn.LogSoftmax(dim=1)

class ResidualGNNs(torch.nn.Module):
    def __init__(self, args, train_dataset, hidden_channels, hidden, num_layers, GNN, k=0.6):
        super().__init__()
        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = hidden_channels
        num_features = train_dataset.num_features
        if args.model == "ChebConv":
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels, K=5))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels, K=5))
        else:
            if num_layers > 0:
                self.convs.append(GNN(num_features, hidden_channels))
                for i in range(0, num_layers - 1):
                    self.convs.append(GNN(hidden_channels, hidden_channels))

        input_dim1 = int(((num_features * num_features) / 2) - (num_features / 2) + (hidden_channels * num_layers))
        input_dim = int(((num_features * num_features) / 2) - (num_features / 2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(hidden_channels * num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden // 2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        for i, xx in enumerate(xs):
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack([t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)] for t in xx])
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)

        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        return softmax(x)

    # ---------- CrossEntropy (logits) variant ----------
    def train_epoch(model, loader, optimizer, device):
        """
        Use this if model.forward returns *logits* (no softmax in forward).
        """
        model.train()
        criterion = CrossEntropyLoss()
        total_loss, total_correct, total_examples = 0.0, 0, 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)  # shape [B, C] logits
            y = batch.y.view(-1).long()  # ensure shape [B]
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * batch.num_graphs
            preds = out.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_examples += batch.num_graphs

        avg_loss = total_loss / max(total_examples, 1)
        acc = total_correct / max(total_examples, 1)
        return avg_loss, acc

    @torch.no_grad()
    def eval_epoch(model, loader, device):
        """
        Eval if model.forward returns *logits*.
        """
        model.eval()
        criterion = CrossEntropyLoss()
        total_loss, total_correct, total_examples = 0.0, 0, 0

        for batch in loader:
            batch = batch.to(device)
            out = model(batch)  # logits
            y = batch.y.view(-1).long()
            loss = criterion(out, y)

            total_loss += float(loss) * batch.num_graphs
            preds = out.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_examples += batch.num_graphs

        avg_loss = total_loss / max(total_examples, 1)
        acc = total_correct / max(total_examples, 1)
        return avg_loss, acc