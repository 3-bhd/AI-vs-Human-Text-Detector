"""
preprocess_utils.py

Utilities for:
- Cleaning raw text
- Computing engineered numeric features.

These are used at inference time to recreate the features for
new, unseen text before applying the saved scaler + SVM model.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Order must match Cleaning.py -> numeric_cols
NUMERIC_COLS = [
    "text_length",
    "word_count",
    "avg_word_len",
    "sentence_count",
    "digit_count",
    "upper_count",
    "punct_count",
    "digit_ratio",
    "upper_ratio",
    "punct_ratio",
    "sent_per_100w",
    "unique_words",
    "type_token_ratio",
    "stopword_count",
    "stopword_ratio",
]


# ---------- Text cleaning (same as Cleaning.py) ----------
def clean_text(t: str) -> str:
    """
    Light normalization used throughout the project:

    - lowercase
    - remove HTML tags
    - remove URLs and emails
    - normalize whitespace
    """
    t = str(t)
    t = t.lower()
    t = re.sub(r"<[^>]+>", " ", t)              # strip HTML tags
    t = re.sub(r"http\S+|www\S+", " ", t)       # URLs
    t = re.sub(r"\S+@\S+\.\S+", " ", t)         # emails
    t = re.sub(r"[\r\n\t]+", " ", t)            # control whitespace
    t = re.sub(r"\s+", " ", t)                  # collapse spaces
    return t.strip()


# ---------- Helpers for numeric features ----------
def _sentence_count(s: str) -> int:
    # naive split by punctuation, same as Cleaning.py
    parts = re.split(r"[.!?]+", s)
    return int(sum(1 for p in parts if p.strip()))


def _count_pattern(s: str, pattern: str) -> int:
    return len(re.findall(pattern, s))


def _safe_div(a, b):
    return (a / b) if b else 0.0


def _unique_word_count(s: str) -> int:
    words = [w for w in s.split() if w]
    return len(set(words))


def _stopword_count(s: str) -> int:
    words = [w for w in s.split() if w]
    return sum(1 for w in words if w in ENGLISH_STOP_WORDS)


# ---------- Main numeric feature function ----------
def compute_numeric_features(text: str) -> np.ndarray:
    """
    Compute ALL engineered numeric features used in Cleaning.py,
    in the SAME order as NUMERIC_COLS.

    Parameters
    ----------
    text : str
        Raw input text (may contain HTML, uppercase, etc.).

    Returns
    -------
    feats : np.ndarray shape (15,)
        [text_length, word_count, avg_word_len, sentence_count,
         digit_count, upper_count, punct_count, digit_ratio,
         upper_ratio, punct_ratio, sent_per_100w, unique_words,
         type_token_ratio, stopword_count, stopword_ratio]
    """
    raw = str(text)
    cleaned = clean_text(raw)

    # core counts
    text_length = len(cleaned)
    words = [w for w in cleaned.split() if w]
    word_count = len(words)
    avg_word_len = _safe_div(text_length, word_count)

    sentence_count = _sentence_count(cleaned)

    digit_count = sum(ch.isdigit() for ch in cleaned)
    upper_count = sum(ch.isupper() for ch in raw)  # from original text
    punct_count = _count_pattern(cleaned, r"[^\w\s]")

    digit_ratio = _safe_div(digit_count, text_length)
    upper_ratio = _safe_div(upper_count, max(1, len(raw)))
    punct_ratio = _safe_div(punct_count, text_length)

    sent_per_100w = 100.0 * _safe_div(sentence_count, word_count)

    unique_words = _unique_word_count(cleaned)
    type_token_ratio = _safe_div(unique_words, word_count)

    stopword_count = _stopword_count(cleaned)
    stopword_ratio = _safe_div(stopword_count, word_count)

    feats = np.array(
        [
            text_length,
            word_count,
            avg_word_len,
            sentence_count,
            digit_count,
            upper_count,
            punct_count,
            digit_ratio,
            upper_ratio,
            punct_ratio,
            sent_per_100w,
            unique_words,
            type_token_ratio,
            stopword_count,
            stopword_ratio,
        ],
        dtype=float,
    )

    return feats
