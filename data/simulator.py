"""
simulator.py
------------
Simulates a real-time SMS stream from a telecom network.

In production, messages would arrive via Kafka from the telecom gateway.
This simulator replicates that behavior by streaming messages from the
dataset to the detection API — one message every few seconds.

Usage:
    python simulator.py
    python simulator.py --speed 1   (1 message per second, faster)
    python simulator.py --speed 5   (1 message every 5 seconds, slower)
    python simulator.py --count 50  (stop after 50 messages)
"""

import argparse
import random
import time

import pandas as pd
import requests

API_URL = "http://localhost:8000/analyze"
DATASET_PATH = "../../data/sms_dataset.csv"


def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} messages from dataset.")
        print(f"Phishing: {df['label'].sum()} | Normal: {(df['label']==0).sum()}\n")
        return df
    except FileNotFoundError:
        print(f"Dataset not found at {path}")
        print("Run data_generator.py first.")
        exit(1)


def send_message(row: dict) -> dict | None:
    payload = {
        "message": row["message"],
        "sender": row.get("sender", "UNKNOWN"),
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        return r.json()
    except requests.exceptions.ConnectionError:
        print("API not reachable. Make sure uvicorn is running on port 8000.")
        return None


def print_result(result: dict, index: int):
    icon = "🚨" if result["prediction"] == "PHISHING" else "✅"
    risk = result["risk_level"]
    conf = int(result["confidence"] * 100)
    sender = result["sender"]
    msg = result["message_preview"][:55] + "..."

    print(f"[{index:04d}] {icon} {result['prediction']:<8} | "
          f"{risk:<6} | {conf:>3}% | {sender:<12} | {msg}")


def main():
    parser = argparse.ArgumentParser(description="SMS stream simulator")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Seconds between messages (default: 2)")
    parser.add_argument("--count", type=int, default=0,
                        help="Number of messages to send (0 = unlimited)")
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--phishing-rate", type=float, default=None,
                        help="Override phishing ratio (0.0-1.0). Default: use dataset ratio.")
    args = parser.parse_args()

    df = load_dataset(args.dataset)

    print("=" * 75)
    print("SMS STREAM SIMULATOR — Sending messages to detection API")
    print("In production this stream would come from Kafka / telecom gateway")
    print("=" * 75)
    print(f"Speed : 1 message every {args.speed}s")
    print(f"Count : {'unlimited' if args.count == 0 else args.count}")
    print("=" * 75)
    print(f"{'#':<6} {'verdict':<12} {'risk':<8} {'conf':>5} {'sender':<14} message")
    print("-" * 75)

    sent = 0
    phishing_sent = 0
    normal_sent = 0

    try:
        while True:
            if args.count > 0 and sent >= args.count:
                break

            # Pick a random row — optionally bias toward phishing
            if args.phishing_rate is not None:
                if random.random() < args.phishing_rate:
                    row = df[df["label"] == 1].sample(1).iloc[0]
                else:
                    row = df[df["label"] == 0].sample(1).iloc[0]
            else:
                row = df.sample(1).iloc[0]

            result = send_message(row.to_dict())

            if result is None:
                print("Stopping simulator — API unreachable.")
                break

            sent += 1
            if result["prediction"] == "PHISHING":
                phishing_sent += 1
            else:
                normal_sent += 1

            print_result(result, sent)
            time.sleep(args.speed)

    except KeyboardInterrupt:
        print("\n" + "=" * 75)
        print("Simulator stopped.")
        print(f"Total sent : {sent}")
        print(f"Phishing   : {phishing_sent} ({phishing_sent/max(sent,1)*100:.1f}%)")
        print(f"Normal     : {normal_sent}")
        print("=" * 75)


if __name__ == "__main__":
    main()