from nilearn import datasets
import numpy as np
from scipy import stats
import torch
from torch_geometric.data import Data


# -----------------------------
# 1) Load the dataset
# -----------------------------

abide = datasets.fetch_abide_pcp(
    data_dir="/path/to/your/ABIDE_pcp",   # <-- your local folder
    derivatives='rois_cc200',
    pipeline='cpac',
    band_pass_filtering=True,
    global_signal_regression=True,
    quality_checked=True
)

# -----------------------------
# 2) Utilities
# -----------------------------
def corr_fisher_z(X):  # X: (T, N) ROI time-series
    C = np.corrcoef(X, rowvar=False)          # (N, N)
    C = np.clip(C, -0.999999, 0.999999)
    Z = np.arctanh(C)                         # Fisher z
    np.fill_diagonal(Z, 0.0)
    return Z

def top_p_sparsify(W, p=0.10, keep_positive=True):
    """Keep top p% absolute (or positive) weights per *whole matrix*."""
    Wc = W.copy()
    if keep_positive:
        Wc[Wc < 0] = 0
    flat = Wc.flatten()
    k = max(1, int(len(flat) * p))
    thr = np.partition(flat, -k)[-k]  # global threshold
    A = (Wc >= thr).astype(float) * Wc
    # symmetrize
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return A

def make_node_features(Z, mode='corr_row'):
    """
    'corr_row' -> node i feature = i-th row of Z (NeuroGraph style)
    'none'     -> no explicit node features (ones)
    """
    N = Z.shape[0]
    if mode == 'corr_row':
        X = Z.copy()
        np.fill_diagonal(X, 0.0)
        return X
    return np.ones((N, 1), dtype=float)

def to_pyg(A, X, y=None, meta=None):
    if Data is None:
        return dict(edge_index=None, edge_weight=None, x=X, y=y, meta=meta)
    # build edge_index/weight from upper triangle
    i, j = np.where(np.triu(A, 1) > 0)
    w = A[i, j]
    # make undirected (i<->j)
    ei = np.vstack([np.hstack([i, j]), np.hstack([j, i])])
    ew = np.hstack([w, w]).astype(np.float32)
    x = torch.tensor(X, dtype=torch.float32)
    data = Data(x=x,
                edge_index=torch.tensor(ei, dtype=torch.long),
                edge_weight=torch.tensor(ew),
                y=None if y is None else torch.tensor([y], dtype=torch.long))
    data.meta = meta
    return data

# -----------------------------
# 3) Build one static graph per subject
# -----------------------------
graphs = []
for path, pheno in zip(abide.rois, abide.phenotypic):
    # Load ROI TS (T x N)
    ts = np.loadtxt(path)  # PCP files are whitespace-CSV
    # 3.1 connectivity (Fisher-z of Pearson corr)
    Z = corr_fisher_z(ts)
    # 3.2 sparsify like NeuroGraph (e.g., top 10% positive edges)
    A = top_p_sparsify(Z, p=0.10, keep_positive=True)
    # 3.3 node features (corr row vectors)
    X = make_node_features(Z, mode='corr_row')
    # 3.4 label: ASD(1)/HC(2) in ABIDE PCP; convert to 0/1
    dx = pheno.get('DX_GROUP')
    y = None if dx is None else int(dx == 1)  # 1=ASD -> 1, 2=HC -> 0
    # 3.5 site/domain
    site = pheno.get('SITE_ID', pheno.get('SITE_NAME', 'NA'))
    graphs.append(to_pyg(A, X, y=y, meta={'site': site, 'age': pheno.get('AGE_AT_SCAN')}))

print(f'Built {len(graphs)} graphs (static). Example meta:', graphs[0].meta if graphs else None)

# -----------------------------
# 4) OPTIONAL: Dynamic graphs (sliding window)
# -----------------------------
def sliding_windows(ts, win=60, step=30):
    T, N = ts.shape
    for start in range(0, max(1, T - win + 1), step):
        yield ts[start:start+win]

dyn_graphs = []
win, step = 60, 30  # tune to your TR/length
for path, pheno in zip(abide.rois, abide.phenotypic):
    ts = np.loadtxt(path)
    for seg in sliding_windows(ts, win=win, step=step):
        Z = corr_fisher_z(seg)
        A = top_p_sparsify(Z, p=0.10, keep_positive=True)
        X = make_node_features(Z, mode='corr_row')
        dx = pheno.get('DX_GROUP')
        y = None if dx is None else int(dx == 1)
        site = pheno.get('SITE_ID', pheno.get('SITE_NAME', 'NA'))
        dyn_graphs.append(to_pyg(A, X, y=y, meta={'site': site, 'dyn': True}))
print(f'Dynamic segments: {len(dyn_graphs)}')