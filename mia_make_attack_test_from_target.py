import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def income_to_bin(y):
    y = y.astype(str).str.strip()
    return y.map({"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1}).astype(int).to_numpy()


@torch.no_grad()
def collect_features(model, X, y, device):
    model.eval().to(device)
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)

    out = model(X_t)

    # support (B,2) -> convert to single logit = logit1 - logit0
    if out.ndim == 2 and out.shape[1] == 2:
        logits = out[:, 1] - out[:, 0]
    else:
        logits = out.squeeze(1) if out.ndim == 2 else out

    p_hat = torch.sigmoid(logits)
    loss = nn.BCEWithLogitsLoss(reduction="none")(logits, y_t)

    return p_hat.cpu().numpy(), loss.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="adulti_buono.csv")
    ap.add_argument("--model_path", type=str, default="adult_mlp_model.pth")
    ap.add_argument("--out_csv", type=str, default="shadows/attack_test_from_target.csv")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # label
    y = income_to_bin(df["income"])

    # features
    X = df.drop(columns=["income"]).copy()

    # EXACTLY like model.py: LabelEncoder fit_transform on whole dataset
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    X = X.astype(np.float32)

    # EXACTLY like model.py: scaler fit_transform on whole dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # EXACTLY like model.py: split with random_state=42, test_size=0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # load model (you saved full model or state_dict?)
    ckpt = torch.load(args.model_path, map_location="cpu")
    if isinstance(ckpt, nn.Module):
        model = ckpt
    else:
        # if you saved only state_dict, import MLP from your model.py
        from model import MLP
        model = MLP(X.shape[1])     # passa input_dim come argomento posizionale

        model.load_state_dict(ckpt)

    # collect features
    p_in, l_in = collect_features(model, X_train, y_train, args.device)
    p_out, l_out = collect_features(model, X_test, y_test, args.device)

    out = pd.concat([
        pd.DataFrame({"p_hat": p_in, "loss": l_in, "member": 1, "true_label": y_train}),
        pd.DataFrame({"p_hat": p_out, "loss": l_out, "member": 0, "true_label": y_test}),
    ], ignore_index=True)

    out.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv, "rows=", len(out))
    print("IN:", len(y_train), "OUT:", len(y_test))


if __name__ == "__main__":
    main()
