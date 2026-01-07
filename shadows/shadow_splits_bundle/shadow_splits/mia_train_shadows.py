import argparse
import glob
import os
import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


# -------------------------
# Label mapping (income -> {0,1})
# -------------------------
def map_income_to_bin(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip()
    mapping = {"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1, "0": 0, "1": 1}
    y = s.map(mapping)
    if y.isna().any():
        vals = s.value_counts().index.tolist()
        if len(vals) != 2:
            raise ValueError(f"Expected binary labels, found: {vals[:10]}")
        y = s.map({vals[0]: 0, vals[1]: 1})
    return y.astype(int).to_numpy()


# -------------------------
# "LabelEncoder-like" per colonna categoriale
# Fit sul train, unknown -> -1
# -------------------------
def fit_cat_mappings(df_train: pd.DataFrame, cat_cols: list[str]) -> dict[str, dict]:
    maps = {}
    for c in cat_cols:
        vals = df_train[c].astype(str).fillna("NA").unique().tolist()
        maps[c] = {v: i for i, v in enumerate(vals)}
    return maps


def transform_with_mappings(df: pd.DataFrame, cat_cols: list[str], num_cols: list[str], maps: dict) -> np.ndarray:
    parts = []

    # categorical -> int
    for c in cat_cols:
        m = maps[c]
        arr = df[c].astype(str).fillna("NA").map(m).fillna(-1).astype(np.int32).to_numpy().reshape(-1, 1)
        parts.append(arr)

    # numeric -> float
    if num_cols:
        num_df = df[num_cols].copy()
        for c in num_cols:
            num_df[c] = pd.to_numeric(num_df[c], errors="coerce")
        num = num_df.fillna(0.0).to_numpy(dtype=np.float32)
        parts.append(num)

    X = np.hstack(parts) if len(parts) > 1 else parts[0]
    return X.astype(np.float32)


# -------------------------
# Torch dataset
# -------------------------
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------
# MLP binary (1 logit)
# -------------------------
class MLPBinary(nn.Module):
    def __init__(self, input_dim: int, h1: int = 128, h2: int = 64, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, h1), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(h1, h2), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(h2, 1)]  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B,)


def train_model(model, loader, device, epochs=10, lr=1e-3, weight_decay=0.0):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(Xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()


@torch.no_grad()
def collect_features(model, loader, device):
    """
    Ritorna:
      - p_hat: sigmoid(logits)
      - loss: BCEWithLogitsLoss per-sample (reduction='none')
    """
    model.eval()
    model.to(device)
    crit_none = nn.BCEWithLogitsLoss(reduction="none")

    p_list, l_list = [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        p = torch.sigmoid(logits)
        l = crit_none(logits, yb)
        p_list.append(p.cpu().numpy())
        l_list.append(l.cpu().numpy())

    return np.concatenate(p_list), np.concatenate(l_list)


def extract_shadow_id(train_path: str) -> int:
    m = re.search(r"shadow_(\d+)_train\.csv$", train_path)
    if not m:
        raise ValueError(f"Cannot parse shadow id from {train_path}")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--label_col", type=str, default="income")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--h1", type=int, default=128)
    ap.add_argument("--h2", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_shadow_models", action="store_true")
    ap.add_argument("--out_attack_csv", type=str, default="attack_train_from_shadows.csv")

    args = ap.parse_args()

    train_files = sorted(glob.glob(os.path.join(args.splits_dir, "shadow_*_train.csv")))
    if not train_files:
        raise FileNotFoundError(f"No shadow_*_train.csv found in: {args.splits_dir}")

    os.makedirs("models", exist_ok=True)

    all_rows = []

    for train_path in train_files:
        sid = extract_shadow_id(train_path)
        out_path = train_path.replace("_train.csv", "_out.csv")
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"Missing out file for shadow {sid:02d}: {out_path}")

        df_tr = pd.read_csv(train_path)
        df_out = pd.read_csv(out_path)

        # labels
        y_tr = map_income_to_bin(df_tr[args.label_col])
        y_out = map_income_to_bin(df_out[args.label_col])

        # features
        Xtr_df = df_tr.drop(columns=[args.label_col])
        Xout_df = df_out.drop(columns=[args.label_col])

        # infer categorical vs numeric (same columns for train/out)
        cat_cols = [c for c in Xtr_df.columns if Xtr_df[c].dtype == "object"]
        num_cols = [c for c in Xtr_df.columns if c not in cat_cols]

        # fit encoders on train only
        maps = fit_cat_mappings(Xtr_df, cat_cols)

        # transform to numeric matrix
        X_tr_raw = transform_with_mappings(Xtr_df, cat_cols, num_cols, maps)
        X_out_raw = transform_with_mappings(Xout_df, cat_cols, num_cols, maps)

        # fit StandardScaler on FULL X_train matrix (coerente con la tua pipeline)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw).astype(np.float32)
        X_out = scaler.transform(X_out_raw).astype(np.float32)

        # loaders
        tr_loader = DataLoader(TabDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
        tr_eval_loader = DataLoader(TabDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=False)
        out_loader = DataLoader(TabDataset(X_out, y_out), batch_size=args.batch_size, shuffle=False)

        # model
        model = MLPBinary(input_dim=X_tr.shape[1], h1=args.h1, h2=args.h2, dropout=args.dropout)

        # train
        train_model(model, tr_loader, device=args.device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)

        if args.save_shadow_models:
            torch.save(
                {"state_dict": model.state_dict(), "cat_cols": cat_cols, "num_cols": num_cols},
                f"models/shadow_{sid:02d}.pth"
            )

        # collect features for attack dataset
        p_in, l_in = collect_features(model, tr_eval_loader, device=args.device)
        p_out, l_out = collect_features(model, out_loader, device=args.device)

        rows_in = pd.DataFrame({
            "shadow_id": sid,
            "p_hat": p_in,
            "loss": l_in,
            "member": 1,
            "true_label": y_tr,
        })
        rows_out = pd.DataFrame({
            "shadow_id": sid,
            "p_hat": p_out,
            "loss": l_out,
            "member": 0,
            "true_label": y_out,
        })

        all_rows.append(rows_in)
        all_rows.append(rows_out)

        print(f"[shadow {sid:02d}] train={len(rows_in)} out={len(rows_out)}")

    attack_df = pd.concat(all_rows, ignore_index=True)
    attack_df.to_csv(args.out_attack_csv, index=False)
    print(f"\nSaved attack dataset: {args.out_attack_csv} (rows={len(attack_df)})")


if __name__ == "__main__":
    main()
