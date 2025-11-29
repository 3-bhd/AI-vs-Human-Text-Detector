# cli_client.py
"""
Console client for the AI vs Human text detector.

Usage:
    1. Make sure the API server is running:
         python api_server.py

    2. Run this client:
         python cli_client.py

    3. Type/paste text when prompted. Type 'q' to quit.

Features:
    - Sends text to /predict endpoint and displays prediction + confidence.
    - Optionally collects user feedback and sends it to /feedback for online learning.
"""

import sys
import requests

API_BASE = "http://localhost:8000"


def call_predict(text: str):
    """Send text to the /predict endpoint and return JSON or None on error."""
    url = f"{API_BASE}/predict"
    try:
        resp = requests.post(url, json={"text": text}, timeout=10)
    except requests.RequestException as e:
        print(f"[ERROR] Failed to reach server: {e}")
        return None

    if resp.status_code != 200:
        try:
            data = resp.json()
        except Exception:
            data = {"error": resp.text}
        print(f"[ERROR] Server returned {resp.status_code}: {data.get('error')}")
        return None

    try:
        return resp.json()
    except ValueError:
        print("[ERROR] Invalid JSON response from server.")
        return None


def call_feedback(text: str, true_label: int):
    """
    Send feedback to /feedback endpoint.

    true_label: 0 (Human-written) or 1 (AI-generated)
    """
    url = f"{API_BASE}/feedback"
    payload = {"text": text, "true_label": true_label}
    try:
        resp = requests.post(url, json=payload, timeout=10)
    except requests.RequestException as e:
        print(f"[ERROR] Failed to send feedback: {e}")
        return

    try:
        data = resp.json()
    except ValueError:
        print("[ERROR] Feedback response was not valid JSON.")
        return

    if resp.status_code != 200:
        print(f"[ERROR] Feedback not accepted ({resp.status_code}): {data}")
    else:
        print("[INFO] Feedback stored successfully.")


def ask_feedback_loop(text: str, prediction_raw: int):
    """
    Ask user if the prediction was correct.
    If not, ask for the true label and send to /feedback.
    """
    while True:
        ans = input("Was this prediction correct? [y]es / [n]o / [s]kip: ").strip().lower()
        if ans in {"y", "yes"}:
            # Optionally: store positive feedback too. For now we skip.
            print("[INFO] OK, not sending feedback (model was correct).")
            return
        elif ans in {"s", "skip", ""}:
            print("[INFO] Skipping feedback for this sample.")
            return
        elif ans in {"n", "no"}:
            break
        else:
            print("Please answer with y / n / s.")

    # Ask user for true label
    while True:
        tl = input("What is the TRUE label? [h]uman / [a]i: ").strip().lower()
        if tl in {"h", "human"}:
            true_label = 0
            break
        elif tl in {"a", "ai"}:
            true_label = 1
            break
        else:
            print("Please answer with 'h' for human or 'a' for AI.")

    call_feedback(text, true_label)


def main():
    print("\n==============================")
    print("  AI vs Human Text Detector")
    print("  CLI Client (talks to API)")
    print("==============================")
    print(f"API base URL: {API_BASE}")
    print("Make sure api_server.py is running!")
    print("Type 'q' on an empty line to quit.\n")

    while True:
        user_text = input("Enter your text (or 'q' to quit):\n> ")

        # Quit logic
        if user_text.strip().lower() in {"q", "quit", "exit"}:
            print("Goodbye!")
            break

        cleaned = user_text.strip()
        if not cleaned:
            print("[INFO] Empty input, nothing to send.")
            continue

        # Call backend
        result = call_predict(cleaned)
        if result is None:
            # Error already printed
            continue

        # Show result
        pred_label = result.get("prediction", "UNKNOWN")
        label_raw = result.get("label_raw", None)
        score = result.get("score", None)
        confidence = result.get("confidence", None)

        print("\n====== PREDICTION RESULT ======")
        print(f"Prediction:     {pred_label}")
        if label_raw is not None:
            print(f"Raw label:      {label_raw}")
        if score is not None:
            print(f"Decision score: {score:.4f}")
        if confidence is not None:
            print(f"Confidence:     {confidence * 100:.2f}%")
        print("================================\n")

        # Ask user for feedback (optional, for online learning)
        if isinstance(label_raw, int):
            ask_feedback_loop(cleaned, label_raw)
        else:
            print("[WARN] label_raw missing; skipping feedback option.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
        sys.exit(0)
