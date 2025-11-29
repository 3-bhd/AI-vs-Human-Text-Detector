# inference_svm.py
"""
Interactive SVM inference script.

Run:
    python inference_svm.py
Then you will be prompted to enter text.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, hstack
from joblib import load
from preprocess_utils import clean_text, compute_numeric_features, NUMERIC_COLS


# ======== PATHS ========
ART = Path("artifacts")
MODEL_DIR = ART / "models"

TFIDF_PATH = ART / "tfidf_vectorizer_small.pkl"
SCALER_PATH = ART / "numeric_scaler.pkl"
MODEL_PATH = MODEL_DIR / "linear_svm_combo.joblib"
TFIDF_SELECTED_TERMS_PATH = ART / "tfidf_selected_terms.csv"

# Same 7 numeric features used in TFIDF.py
NUMERIC_KEEP = [
    "stopword_ratio",
    "stopword_count",
    "sentence_count",
    "unique_words",
    "punct_ratio",
    "type_token_ratio",
    "avg_word_len",
]

NUMERIC_WEIGHT = 0.30  # same weighting as TFIDF.py


# ======== LOAD ARTIFACTS ========
print("Loading TF-IDF vectorizer...")
vectorizer = load(TFIDF_PATH)

print("Loading numeric scaler...")
scaler = load(SCALER_PATH)

print("Loading SVM model...")
clf = load(MODEL_PATH)

# Map numeric names to indices (15 numeric features)
numeric_index_map = {name: i for i, name in enumerate(NUMERIC_COLS)}
selected_numeric_indices = [numeric_index_map[f] for f in NUMERIC_KEEP]


# --- Load χ²-selected TF-IDF terms ---
print("Loading χ²-selected TF-IDF terms...")
selected_terms_df = pd.read_csv(TFIDF_SELECTED_TERMS_PATH)
selected_terms = selected_terms_df["term"].astype(str).tolist()
print(f"Selected terms: {len(selected_terms)}")

# Build mapping term -> index
vocab = vectorizer.vocabulary_
tfidf_selected_indices = []
missing_terms = []

for term in selected_terms:
    idx = vocab.get(term)
    if idx is None:
        missing_terms.append(term)
    else:
        tfidf_selected_indices.append(idx)

if missing_terms:
    print(f"WARNING: {len(missing_terms)} missing terms from vocabulary.")

print(f"Using {len(tfidf_selected_indices)} TF-IDF indices after χ² feature alignment.")


# ======== FEATURE PIPELINE ========
def featurize_texts(text_list):
    """
    Convert texts → final combined features:
    TF-IDF (20000 dims) + Numeric (7 dims)
    """
    cleaned = [clean_text(t) for t in text_list]

    # full TF-IDF vector
    X_tfidf_full = vectorizer.transform(cleaned)

    # χ²-selected subset
    X_tfidf = X_tfidf_full[:, tfidf_selected_indices]

    # numeric features
    raw_numeric = np.vstack([compute_numeric_features(t) for t in cleaned])
    scaled_numeric = scaler.transform(raw_numeric)
    selected_numeric = scaled_numeric[:, selected_numeric_indices]
    selected_numeric *= NUMERIC_WEIGHT

    # sparse numeric
    X_num = csr_matrix(selected_numeric)

    # final feature vector
    return hstack([X_tfidf, X_num], format="csr")


def predict_text(text):
    """
    Returns:
      pred        -> 0/1
      label       -> human-readable label
      score       -> SVM margin
      confidence  -> logistic(score)
    """
    X = featurize_texts([text])

    # margin score
    score = clf.decision_function(X)[0]

    # predicted label
    pred = clf.predict(X)[0]

    # convert margin → pseudo-probability
    prob_ai = 1.0 / (1.0 + np.exp(-score))
    confidence = prob_ai if pred == 1 else (1 - prob_ai)

    label = "AI-generated" if pred == 1 else "Human-written"
    return pred, label, score, confidence


# ======== INTERACTIVE MODE ========
if __name__ == "__main__":
    # You can optionally put a tiny test here or just pass
    text = "This is a test."
    print(predict_text(text))

    # user_text = input("Enter your text:\n\n")

    # pred, label, score, confidence = predict_text(user_text)

    # print("\n====== PREDICTION ======")
    # print("Raw label:", pred)
    # print("Interpretation:", label)
    # print(f"Decision score: {score:.4f}  (positive = AI, negative = Human)")
    # print(f"Confidence: {confidence * 100:.2f}%")
    # print("========================\n")
