import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

train_df = pd.read_csv("shadows/attack_train_from_shadows.csv")
test_df  = pd.read_csv("shadows/attack_test_from_target.csv")

features = ["p_hat", "loss"]

all_scores = []
all_true = []
all_pred = []

for c in sorted(train_df["true_label"].unique()):
    tr = train_df[train_df["true_label"] == c]
    te = test_df[test_df["true_label"] == c]

    Xtr = tr[features].to_numpy(dtype=np.float32)
    ytr = tr["member"].to_numpy(dtype=int)

    Xte = te[features].to_numpy(dtype=np.float32)
    yte = te["member"].to_numpy(dtype=int)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    clf.fit(Xtr, ytr)

    score = clf.predict_proba(Xte)[:, 1]
    pred = (score >= 0.5).astype(int)

    print(f"[class={c}] AUC:", roc_auc_score(yte, score))

    all_scores.append(score)
    all_true.append(yte)
    all_pred.append(pred)

all_scores = np.concatenate(all_scores)
all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)

print("\n=== FINAL (per-class aggregated) ===")
print("AUC:", roc_auc_score(all_true, all_scores))
print("ACC:", accuracy_score(all_true, all_pred))
print(classification_report(all_true, all_pred, digits=4))
