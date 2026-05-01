"""
detector.py
-----------
Loads the trained model and scores new SMS messages in real-time.
Can be used standalone (CLI) or imported by the API.

Usage:
    python detector.py --message "Click here to claim your prize: http://win.xyz/abc"
    python detector.py --file data/sms_dataset.csv --output results/scored.csv
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

# ── Feature extraction (same logic as data_generator.py) ─────────────────────

def extract_features(message: str) -> dict:
    text = message.lower()

    has_url = int(any(p in text for p in ["http://", "https://", "bit.ly", "tinyurl"]))

    suspicious_tlds = [".xyz", ".tk", ".ml", ".ga", ".cf", ".ru", ".info"]
    has_suspicious_tld = int(any(tld in text for tld in suspicious_tlds))

    urgency_words = ["urgent", "immediately", "now", "expires", "suspended",
                     "locked", "blocked", "final", "warning", "alert", "action required"]
    has_urgency_words = int(any(w in text for w in urgency_words))

    prize_words = ["won", "winner", "prize", "congratulations", "free",
                   "gift", "lottery", "selected", "reward", "claim"]
    has_prize_words = int(any(w in text for w in prize_words))

    shorteners = ["bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly"]
    url_shortener_present = int(any(s in text for s in shorteners))

    special_char_count = sum(1 for c in message if c in "!@#$%^&*()_+=[]{}|;:,.<>?")
    message_length = len(message)
    digit_ratio = round(sum(1 for c in message if c.isdigit()) / max(len(message), 1), 4)

    return {
        "has_url": has_url,
        "has_suspicious_tld": has_suspicious_tld,
        "has_urgency_words": has_urgency_words,
        "has_prize_words": has_prize_words,
        "url_shortener_present": url_shortener_present,
        "special_char_count": special_char_count,
        "message_length": message_length,
        "digit_ratio": digit_ratio,
    }


FEATURE_COLS = [
    "has_url", "has_suspicious_tld", "has_urgency_words", "has_prize_words",
    "url_shortener_present", "special_char_count", "message_length", "digit_ratio",
]

# ── Load model ────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


# ── Score a single message ────────────────────────────────────────────────────

def score_message(pipeline, message: str, sender: str = "UNKNOWN") -> dict:
    features = extract_features(message)
    X = pd.DataFrame([features])[FEATURE_COLS]

    prediction = pipeline.predict(X)[0]
    confidence = pipeline.predict_proba(X)[0][1]  # probability of being phishing

    # Risk level based on confidence
    if confidence >= 0.85:
        risk = "HIGH"
    elif confidence >= 0.50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "sender": sender,
        "message": message[:80] + ("..." if len(message) > 80 else ""),
        "prediction": "PHISHING" if prediction == 1 else "NORMAL",
        "confidence": round(float(confidence), 4),
        "risk_level": risk,
        "features": features,
    }


# ── Score a batch of messages from CSV ───────────────────────────────────────

def score_batch(pipeline, input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    print(f"Scoring {len(df)} messages...")

    features_list = df["message"].apply(extract_features).tolist()
    features_df = pd.DataFrame(features_list)[FEATURE_COLS]

    df["prediction"] = pipeline.predict(features_df)
    df["confidence"] = pipeline.predict_proba(features_df)[:, 1].round(4)
    df["risk_level"] = df["confidence"].apply(
        lambda c: "HIGH" if c >= 0.85 else ("MEDIUM" if c >= 0.50 else "LOW")
    )
    df["predicted_label"] = df["prediction"].map({1: "PHISHING", 0: "NORMAL"})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    total = len(df)
    flagged = (df["prediction"] == 1).sum()
    high_risk = (df["risk_level"] == "HIGH").sum()

    print(f"\nBatch Scoring Summary:")
    print(f"  Total messages : {total}")
    print(f"  Flagged        : {flagged} ({flagged/total*100:.1f}%)")
    print(f"  High risk      : {high_risk}")
    print(f"  Results saved  : {output_path}")

    return df


# ── Pretty print single result ────────────────────────────────────────────────

def print_result(result: dict):
    verdict_icon = "🚨" if result["prediction"] == "PHISHING" else "✅"
    risk_color = {"HIGH": "❗", "MEDIUM": "⚠️", "LOW": "✔️"}

    print("\n" + "=" * 55)
    print(f"  VERDICT: {verdict_icon}  {result['prediction']}")
    print(f"  Risk level  : {risk_color[result['risk_level']]}  {result['risk_level']}")
    print(f"  Confidence  : {result['confidence']*100:.1f}%")
    print(f"  Sender      : {result['sender']}")
    print(f"  Message     : {result['message']}")
    print("-" * 55)
    print("  Detected signals:")
    f = result["features"]
    if f["has_url"]:
        print("    • Contains URL")
    if f["has_suspicious_tld"]:
        print("    • Suspicious domain extension (.xyz, .tk, etc.)")
    if f["has_urgency_words"]:
        print("    • Urgency language detected")
    if f["has_prize_words"]:
        print("    • Prize/reward language detected")
    if f["url_shortener_present"]:
        print("    • URL shortener detected (bit.ly, tinyurl, etc.)")
    print("=" * 55)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SMS Phishing Detector")
    parser.add_argument("--model", default="models/phishing_model.pkl")
    parser.add_argument("--message", type=str, help="Single SMS message to score")
    parser.add_argument("--sender", type=str, default="UNKNOWN")
    parser.add_argument("--file", type=str, help="CSV file to score in batch")
    parser.add_argument("--output", type=str, default="results/scored.csv")
    args = parser.parse_args()

    pipeline = load_model(args.model)
    print(f"Model loaded: {args.model}")

    if args.message:
        result = score_message(pipeline, args.message, sender=args.sender)
        print_result(result)

    elif args.file:
        score_batch(pipeline, args.file, args.output)

    else:
        # Demo mode — score a few example messages
        print("\nRunning demo with sample messages...\n")
        examples = [
            ("URGENT: Your account suspended. Verify: http://mci-secure.xyz/abc123", "INFO"),
            ("Your OTP code is 847291. Valid for 5 minutes. Do not share.", "MCI"),
            ("Congratulations! You won a prize. Claim here: http://bit.ly/xyz99", "PRIZE"),
            ("Hi Sara, your appointment is confirmed for 2026-05-10 at 14:00.", "Clinic"),
            ("ALERT: Suspicious login detected. Confirm identity: http://secure-login.tk/verify", "ALERT"),
            ("Your order #543210 has been shipped. Expected delivery: 2026-05-03.", "Digikala"),
        ]

        for msg, sender in examples:
            result = score_message(pipeline, msg, sender=sender)
            print_result(result)


if __name__ == "__main__":
    main()