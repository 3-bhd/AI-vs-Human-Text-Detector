# train_linear_svm.py
"""
Final Linear SVM training script.

- Loads precomputed feature matrices (combo or tfidf).
- Trains the FINAL LinearSVC with Phase IV best hyperparameters.
- Evaluates on held-out test set.
- Saves the trained model to artifacts/models/linear_svm_combo.joblib
  for use by inference_svm.py.
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, issparse

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from joblib import dump

# ---------- I/O helpers & paths ----------

ART = Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)

OUT = ART / "phase3_results"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_DIR = ART / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _try_load_npz(name: str):
    """Load .npz from current dir or artifacts/."""
    p1 = Path(name)
    p2 = ART / name
    if p1.exists():
        return load_npz(p1)
    if p2.exists():
        return load_npz(p2)
    raise FileNotFoundError(f"Could not find {name} in '.' or 'artifacts/'")


def _try_load_npy(name: str):
    """Load .npy from current dir or artifacts/."""
    p1 = Path(name)
    p2 = ART / name
    if p1.exists():
        return np.load(p1)
    if p2.exists():
        return np.load(p2)
    raise FileNotFoundError(f"Could not find {name} in '.' or 'artifacts/'")


# ---------- Load training & test features ----------

# Prefer hybrid combo features; fall back to pure TF-IDF if needed
try:
    Xtr = _try_load_npz("X_train_combo_small.npz")
    Xte = _try_load_npz("X_test_combo_small.npz")
    feature_tag = "combo"
except FileNotFoundError:
    Xtr = _try_load_npz("X_train_tfidf_small.npz")
    Xte = _try_load_npz("X_test_tfidf_small.npz")
    feature_tag = "tfidf"

y_tr = _try_load_npy("y_train.npy")
y_te = _try_load_npy("y_test.npy")

assert issparse(Xtr) and issparse(Xte), "Expecting sparse CSR/CSC matrices."
assert Xtr.shape[0] == y_tr.shape[0]
assert Xte.shape[0] == y_te.shape[0]

print(f"Feature space: {feature_tag}")
print(f"Train shape: {Xtr.shape}")
print(f"Test  shape: {Xte.shape}")

# ---------- Train FINAL Linear SVM (no GridSearch) ----------

# Best hyperparameters from Phase IV:
# C = 10.0, loss = "squared_hinge", class_weight = None, max_iter = 10000
clf = LinearSVC(
    C=10.0,
    loss="squared_hinge",
    class_weight=None,
    max_iter=10_000,
    random_state=42,
)

print("\nTraining final LinearSVC model with fixed tuned hyperparameters...")
clf.fit(Xtr, y_tr)
print("Training complete.")

# ---------- Evaluate on held-out test set ----------

print("\nEvaluating on held-out test set...")
y_pred = clf.predict(Xte)

# Use decision_function for ranking metrics (ROC AUC, PR AUC)
y_score = None
try:
    df = clf.decision_function(Xte)
    y_score = df if df.ndim == 1 else df[:, 1]
except Exception:
    y_score = None

acc = accuracy_score(y_te, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_te, y_pred, average="macro", zero_division=0
)

metrics = {
    "model_type": "LinearSVC",
    "feature_tag": feature_tag,
    # fixed hyperparameters
    "C": 10.0,
    "loss": "squared_hinge",
    "class_weight": None,
    "max_iter": 10_000,
    # test metrics
    "test_accuracy": float(acc),
    "test_precision_macro": float(prec),
    "test_recall_macro": float(rec),
    "test_f1_macro": float(f1),
}

if set(np.unique(y_te)) <= {0, 1} and y_score is not None:
    try:
        metrics["test_roc_auc"] = float(roc_auc_score(y_te, y_score))
    except Exception:
        pass
    try:
        metrics["test_pr_auc"] = float(average_precision_score(y_te, y_score))
    except Exception:
        pass

cm = confusion_matrix(y_te, y_pred)
rep = classification_report(
    y_te, y_pred, output_dict=True, zero_division=0
)

print(f"Test accuracy: {acc:.6f}")
print(f"Test macro-F1: {f1:.6f}")

# ---------- Persist results (metrics, confusion matrix, report) ----------

tag = f"{feature_tag}__svm_linear_final"

pd.DataFrame(
    cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]
).to_csv(OUT / f"{tag}__confusion_matrix.csv")

pd.DataFrame(rep).to_csv(OUT / f"{tag}__class_report.csv")

with open(OUT / f"{tag}__metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved evaluation artifacts:")
print(" -", OUT / f"{tag}__metrics.json")
print(" -", OUT / f"{tag}__confusion_matrix.csv")
print(" -", OUT / f"{tag}__class_report.csv")

# ---------- Save trained model for inference ----------

model_path = MODEL_DIR / "linear_svm_combo.joblib"
dump(clf, model_path)
print("\nSaved final Linear SVM model to:")
print(" -", model_path)

print("\nDone.")
