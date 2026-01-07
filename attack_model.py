import argparse
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="shadows/attack_train_from_shadows.csv")
    ap.add_argument("--test_csv", type=str, default="shadows/attack_test_from_target.csv")
    ap.add_argument("--features", nargs="+", default=["p_hat", "loss"])
    args = ap.parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    X_train = train_df[args.features].to_numpy(dtype=np.float32)
    y_train = train_df["member"].to_numpy(dtype=int)

    X_test = test_df[args.features].to_numpy(dtype=np.float32)
    y_test = test_df["member"].to_numpy(dtype=int)

    # Logistic Regression (simple + robust)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)

    # predictions
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)

    print("=== Attack model evaluation on TARGET ===")
    print("Features:", args.features)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}\n")

    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_test, y_pred), "\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()
