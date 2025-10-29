# NeuroGraph Domain Adaptation - Data Module

This project provides a convenience wrapper around `torch_geometric.datasets.NeuroGraphDataset` for fMRI graph data.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from data import NeuroGraphDataModule

# Point `root` to the directory where NeuroGraphDataset is (will be) stored.
dm = NeuroGraphDataModule(root="/path/to/neurograph", batch_size=16, num_workers=4)

dm.setup()
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
test_loader = dm.test_dataloader()

print("Domain counts:", dm.domain_counts())
```
