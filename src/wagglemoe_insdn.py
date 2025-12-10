# ---- WaggleMoE-TabTransformer on InSDN (original CSVs) ----
# Repo version: assumes CSV files under data/InSDN_DatasetCSV/

import os, re, math, numpy as np, pandas as pd, warnings
from copy import deepcopy
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import sparse

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ===================== 1) Paths =====================
# repo-relative base path
BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "InSDN_DatasetCSV")
BASE = os.path.abspath(BASE)

FILES = {
    "metasploitable-2.csv": 1,  # Attack
    "Normal_data.csv": 0,       # Normal
    "OVS.csv": 1                # Attack/mixed
}

# ===================== 2) Load & Clean =====================
def load_original_insdn():
    dfs = []
    for fname, attack_flag in FILES.items():
        path = os.path.join(BASE, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

        # Fix Label column if needed
        if 'Label' not in df.columns:
            cand = [c for c in df.columns if c.lower() in ['label','labels','class','attack']]
            if cand: df = df.rename(columns={cand[0]: 'Label'})
        if 'Label' not in df.columns and attack_flag==0:
            df['Label'] = 'Normal'
        elif 'Label' not in df.columns and attack_flag==1:
            df['Label'] = 'Attack'

        dfs.append(df)

    raw = pd.concat(dfs, axis=0, ignore_index=True)

    # Drop identifiers / leakage
    patterns = ['Flow ID','Src IP','Src Port','Dst IP','Dst Port','Timestamp','Unnamed','^id$','^pid$']
    pat = re.compile('|'.join(patterns), re.IGNORECASE)
    drop_cols = [c for c in raw.columns if pat.search(c)]
    raw = raw.drop(columns=drop_cols, errors='ignore')

    # Drop constant / empty columns
    nun = raw.nunique(dropna=False)
    const_cols = nun[nun<=1].index.tolist()
    raw = raw.drop(columns=const_cols, errors='ignore')

    # Binary: Attack vs Normal
    y_binary = (~raw['Label'].astype(str).str.lower().str.strip().eq('normal')).astype(int)
    X = raw.drop(columns=['Label'], errors='ignore')
    return X, y_binary, raw

# ===================== 3) Preprocess (Impute+Scale & OneHot) =====================
def preprocess_insdn():
    X, y_bin, raw_all = load_original_insdn()
    print("Original merged shape:", raw_all.shape)
    print("X shape:", X.shape, "| Attack rate:", y_bin.mean().round(4))

    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_tf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=True))
    ])

    cat_tf = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    pre = ColumnTransformer([
        ('num', num_tf, num_cols),
        ('cat', cat_tf, cat_cols)
    ])

    # Train/Val/Test split (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_bin, test_size=0.30, random_state=SEED, stratify=y_bin
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    # Fit preprocessor on train only
    Xtr = pre.fit_transform(X_train)
    Xv  = pre.transform(X_val)
    Xte = pre.transform(X_test)

    return Xtr, Xv, Xte, y_train, y_val, y_test

# ===================== 4) Feature Selection (MI) =====================
K_FEATURES = 64  # try 64/128 depending on memory

def to_dense_safe(M):
    return M.todense() if sparse.issparse(M) else np.asarray(M)

def select_top_features(Xtr, Xv, Xte, y_train):
    use_sample_for_mi = True
    mi_sample_n = min(100000, Xtr.shape[0])

    if use_sample_for_mi and Xtr.shape[0] > mi_sample_n:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(Xtr.shape[0], size=mi_sample_n, replace=False)
        X_mi = to_dense_safe(Xtr[idx])
        y_mi = y_train.values[idx]
    else:
        X_mi = to_dense_safe(Xtr)
        y_mi = y_train.values

    mi = mutual_info_classif(np.asarray(X_mi), y_mi, discrete_features=False, random_state=SEED)
    mi_idx = np.argsort(-mi)[:K_FEATURES]

    def select_k(M, idx=mi_idx):
        return M[:, idx] if sparse.issparse(M) else np.asarray(M)[:, idx]

    Xtr_k = select_k(Xtr); Xv_k = select_k(Xv); Xte_k = select_k(Xte)
    INPUT_DIM = Xtr_k.shape[1]
    print("Selected features:", INPUT_DIM)
    return Xtr_k, Xv_k, Xte_k, INPUT_DIM

# ===================== 5) PyTorch Dataset & DataLoader =====================
class TabDataset(Dataset):
    def __init__(self, X, y):
        if sparse.issparse(X): X = X.toarray()
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# ===================== 6) WaggleMoE-TabTransformer (Model) =====================
class TabBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=8, ff=256, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ff, d_model))
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        # x: [B, T=1, d]
        h = self.ln1(x)
        a,_ = self.attn(h,h,h,need_weights=False)
        x = x + self.drop(a)
        h = self.ln2(x)
        x = x + self.drop(self.ffn(h))
        return x

class WaggleGate(nn.Module):
    """
    Waggle gate with epsilon-annealing exploration and load-balance regularization.
    """
    def __init__(self, d_in, n_experts=6, k=2):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.router = nn.Linear(d_in, n_experts)
        hidden = 256
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_in))
            for _ in range(n_experts)
        ])

    def forward(self, x, epsilon=0.1):
        # x: [B, d]
        logits = self.router(x)                  # [B, E]
        probs = F.softmax(logits, dim=-1)        # [B, E]
        B, E = probs.shape
        if epsilon > 0:
            probs = (1-epsilon)*probs + epsilon*(1.0/E)

        topk = torch.topk(probs, k=self.k, dim=-1)   # values, indices
        idx = topk.indices                            # [B, k]
        wts = topk.values                             # [B, k]

        # Load-balance loss: encourage near-uniform usage
        load = probs.mean(0)                          # [E]
        aux = (load * torch.log(load * E + 1e-9)).sum() / math.log(E+1e-9)

        out = torch.zeros_like(x)
        for j in range(self.k):
            ex_idx = idx[:, j]
            part = torch.stack([ self.experts[e](x[i]) for i,e in enumerate(ex_idx.tolist()) ], dim=0)
            out = out + wts[:, j:j+1]*part

        return out, aux

class WaggleMoETabTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, depth=3, n_heads=8, dropout=0.1,
                 n_experts=6, top_k=2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([TabBlock(d_model=d_model, n_heads=n_heads, ff=4*d_model, dropout=dropout)
                                     for _ in range(depth)])
        self.moe_gate = WaggleGate(d_model, n_experts=n_experts, k=top_k)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x, epsilon=0.1):
        h = self.embed(x).unsqueeze(1)  # [B, 1, d]
        for blk in self.blocks:
            h = blk(h)
        z = h.squeeze(1)                # [B, d]
        moe_out, aux = self.moe_gate(z, epsilon=epsilon)
        z = z + moe_out                 # residual
        logit = self.head(z).squeeze(-1)
        return logit, aux

# ===================== 7) Train / Eval loops =====================
def train_one_epoch(model, loader, opt, device, epsilon, lb_lambda=0.1):
    model.train()
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logit, aux = model(xb, epsilon=epsilon)
        loss = F.binary_cross_entropy_with_logits(logit, yb) + lb_lambda*aux
        loss.backward()
        opt.step()
        tot += loss.item()*xb.size(0); n += xb.size(0)
    return tot/n

@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs, targs = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logit, _ = model(xb, epsilon=0.0)
        p = torch.sigmoid(logit).detach().cpu().numpy()
        probs.append(p); targs.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(targs)

def tune_threshold(p_val, y_val, mode='f1'):
    best_t, best_s = 0.5, -1
    for t in np.linspace(0.05, 0.95, 181):
        y_hat = (p_val >= t).astype(int)
        s = recall_score(y_val, y_hat, zero_division=0) if mode=='recall' else f1_score(y_val, y_hat, zero_division=0)
        if s > best_s: best_s, best_t = s, t
    return best_t, best_s

# ===================== 8) Main script =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    Xtr, Xv, Xte, y_train, y_val, y_test = preprocess_insdn()
    Xtr_k, Xv_k, Xte_k, INPUT_DIM = select_top_features(Xtr, Xv, Xte, y_train)

    BATCH = 1024
    train_ds = TabDataset(Xtr_k, y_train.values)
    val_ds   = TabDataset(Xv_k,  y_val.values)
    test_ds  = TabDataset(Xte_k, y_test.values)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model = WaggleMoETabTransformer(
        input_dim=INPUT_DIM, d_model=128, depth=3, n_heads=8, dropout=0.1,
        n_experts=6, top_k=2
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    EPOCHS = 15
    EPS_START, EPS_END = 0.2, 0.02

    def epsilon_at(ep):
        return EPS_END + (EPS_START-EPS_END)*max(0, (EPOCHS-1-ep))/(EPOCHS-1)

    best_val = 1e9
    patience, bad = 4, 0
    best_state = None

    for ep in range(EPOCHS):
        eps = float(epsilon_at(ep))
        tr_loss = train_one_epoch(model, train_loader, opt, device, epsilon=eps, lb_lambda=0.1)
        pv, yv = predict_proba(model, val_loader, device)
        val_loss = F.binary_cross_entropy(torch.tensor(pv), torch.tensor(yv, dtype=torch.float32)).item()
        print(f"Epoch {ep+1}/{EPOCHS} | eps={eps:.3f} | train_loss={tr_loss:.4f} | val_bce={val_loss:.4f}")
        if val_loss < best_val - 1e-4:
            best_val = val_loss; bad = 0; best_state = deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping."); break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Threshold tuning & Evaluation
    TUNE_MODE = 'f1'
    p_val, y_val_np = predict_proba(model, val_loader, device)
    t_star, s_star = tune_threshold(p_val, y_val_np, mode=TUNE_MODE)
    print(f"\nBest threshold on Val ({TUNE_MODE.upper()}): t={t_star:.2f}, score={s_star:.4f}")

    p_test, y_test_np = predict_proba(model, test_loader, device)
    y_hat = (p_test >= t_star).astype(int)

    print("\n=== Test Evaluation (Binary: Attack vs Normal) ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test_np, y_hat))
    print("\nClassification Report:\n", classification_report(y_test_np, y_hat, digits=4))
    print("Accuracy:", accuracy_score(y_test_np, y_hat))
    print("F1:", f1_score(y_test_np, y_hat))
    try:
        print("ROC-AUC:", roc_auc_score(y_test_np, p_test))
    except Exception as e:
        print("ROC-AUC error:", e)

if __name__ == "__main__":
    main()
