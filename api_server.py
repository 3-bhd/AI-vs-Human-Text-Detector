# api_server.py
"""
Flask backend API for the AI vs Human text detector.

Endpoints
---------
GET  /health
    → simple status + model name.

POST /predict
    JSON: { "text": "<string>" }
    → { "prediction": "AI-generated"|"Human-written",
         "label_raw": 0|1,
         "score": float,
         "confidence": float }

POST /feedback
    JSON: { "text": "<string>", "true_label": 0|1|"AI"|"Human"|"AI-generated"|"Human-written" }
    → stores feedback in feedback_store.csv (used later for retraining).
"""

from flask import Flask, request, jsonify
from pathlib import Path
from datetime import datetime
import csv

from inference_svm import predict_text  # uses vectorizer, scaler, SVM, etc.

# ---------- App + paths ----------

from pathlib import Path
app = Flask(__name__, static_folder="static")

PROJECT_ROOT = Path(__file__).resolve().parent
FEEDBACK_PATH = PROJECT_ROOT / "feedback_store.csv"


# ---------- Helpers ----------

def _normalize_true_label(raw):
    """
    Convert various user labels into 0/1.

    Accepts:
      0, 1,
      "0", "1",
      "ai", "AI", "AI-generated",
      "human", "Human", "Human-written"

    Returns:
      0 or 1, or raises ValueError on invalid input.
    """
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return int(raw)

    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"0", "human", "human-written"}:
            return 0
        if s in {"1", "ai", "ai-generated"}:
            return 1

    raise ValueError(f"Invalid true_label: {raw!r}")


def _append_feedback(text: str, true_label: int):
    """
    Append one row of feedback to feedback_store.csv
    with columns: timestamp, true_label, text
    """
    new_file = not FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "true_label", "text"])
        writer.writerow(
            [datetime.utcnow().isoformat(timespec="seconds"), true_label, text]
        )


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    # Serve static/index.html
    return app.send_static_file("index.html")

@app.route("/health", methods=["GET"])
def health():
    """
    Lightweight health check endpoint.
    """
    return jsonify(
        {
            "status": "ok",
            "model": "linear_svm_combo",
            "message": "Backend is running.",
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Expects JSON:
      { "text": "<user text>" }
    """
    # 1) Validate JSON
    if not request.is_json:
        return (
            jsonify({"error": "Invalid request: expected JSON body."}),
            400,
        )

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON."}), 400

    text = data.get("text", None)

    # 2) Validate text field
    if text is None:
        return jsonify({"error": "Invalid input: 'text' field is required."}), 400

    if not isinstance(text, str):
        return jsonify({"error": "Invalid input: 'text' must be a string."}), 400

    cleaned = text.strip()
    if len(cleaned) == 0:
        return jsonify({"error": "Invalid input: 'text' cannot be empty."}), 400

    # Optional: minimum length check
    if len(cleaned.split()) < 3:
        return (
            jsonify(
                {
                    "error": "Input too short: please provide at least a few words."
                }
            ),
            400,
        )

    # 3) Run prediction via inference_svm
    try:
        pred_raw, label_human, score, confidence = predict_text(cleaned)
    except Exception as e:
        # Log error server-side in real deployment; here we just return generic error
        return (
            jsonify(
                {
                    "error": "Internal error while running prediction.",
                    "details": str(e),
                }
            ),
            500,
        )

    # 4) Build response
    resp = {
        "prediction": label_human,      # "AI-generated" or "Human-written"
        "label_raw": int(pred_raw),     # 1 or 0
        "score": float(score),          # SVM margin
        "confidence": float(confidence) # 0.0–1.0
    }

    return jsonify(resp), 200


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Feedback endpoint for online learning.

    Expects JSON:
      {
        "text": "<same text the model saw>",
        "true_label": 0|1|"AI"|"Human"|...
      }

    Stores feedback in feedback_store.csv for future retraining.
    """
    if not request.is_json:
        return (
            jsonify({"error": "Invalid request: expected JSON body."}),
            400,
        )

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Invalid JSON."}), 400

    text = data.get("text", None)
    true_label_raw = data.get("true_label", None)

    if text is None or not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be non-empty."}), 400

    if true_label_raw is None:
        return jsonify({"error": "Field 'true_label' is required."}), 400

    try:
        true_label = _normalize_true_label(true_label_raw)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Append to CSV
    try:
        _append_feedback(text.strip(), true_label)
    except Exception as e:
        return (
            jsonify(
                {"error": "Failed to store feedback.", "details": str(e)}
            ),
            500,
        )

    return jsonify({"status": "ok", "message": "Feedback stored."}), 200


# ---------- Main ----------

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

