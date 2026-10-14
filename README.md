# NeuroGraph

Graph domain adaptation for fMRI connectome classification (ABIDE, autism vs. control) across acquisition sites.

## GJDOT

`gjdot_pipeline_ABIDE.py` runs GJDOT (JDOT block-coordinate descent with a GIN classifier, ported from the gjdot repository at commit `26c0075`) between ABIDE sites. Each subject is a graph over the CC200 atlas ROIs; the ground cost between two subjects is either the FGW distance or the Frobenius distance of their Fisher-z connectivity matrices.

```
GJDOT/
  gjdot_abide.py        ABIDE download (nilearn) and connectome-graph construction
  gjdot_costs.py        FGW / Frobenius graph cost matrices and caching
  gjdot_ot.py           OT solvers, label losses, JDOT objective terms
  gjdot_classifiers.py  JDOT classifier interface (GIN)
  gjdot_models.py       GIN training: source-only baseline and JDOT refits
  gjdot_gnn.py          GNN architecture
  gjdot_logging.py      MLflow metric and plot logging
  gjdot_utils.py        seeding, parameter grid, domain sampling
gjdot_pipeline_ABIDE.py experiment: parameter grid at the top, one MLflow run per combination
```

Run from the repository root:

```
pip install -r requirements.txt
python gjdot_pipeline_ABIDE.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
python -m pytest tests
```

ABIDE time series are downloaded to `data/nilearn_data/` on first use (only the sites in `SITE_GROUPS` that appear in the grid). Cost matrices are cached in `cost_matrices/`.
