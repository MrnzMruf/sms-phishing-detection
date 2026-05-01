"""
data_generator.py
-----------------
Generates a realistic synthetic SMS dataset for phishing detection.
Produces ~10,000 labeled messages (normal + phishing) saved as CSV.

Usage:
    python data_generator.py
    python data_generator.py --output data/sms_dataset.csv --samples 20000
"""

import random
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

# ── Phishing templates ────────────────────────────────────────────────────────
PHISHING_TEMPLATES = [
    "URGENT: Your account has been suspended. Verify now: {url}",
    "Congratulations! You won a prize. Claim here: {url}",
    "Your package could not be delivered. Track here: {url}",
    "Security alert: unusual login detected. Confirm identity: {url}",
    "Your bank account is locked. Unlock now: {url}",
    "FREE gift waiting for you! Click: {url}",
    "Action required: update your payment info at {url}",
    "You have a pending refund of {amount} IRR. Claim: {url}",
    "Dear customer, your subscription expires today. Renew: {url}",
    "WARNING: Your SIM will be deactivated. Verify: {url}",
    "Tax refund available: {amount} IRR. Apply now: {url}",
    "Your account shows suspicious activity. Review: {url}",
    "Limited offer: get {discount}% off. Only today: {url}",
    "Your password was changed. If not you, click: {url}",
    "Lottery winner! You have been selected. Collect: {url}",
    "MCI: Your line will be blocked. Confirm details: {url}",
    "Final notice: invoice overdue. Pay now to avoid suspension: {url}",
    "Your delivery is on hold. Confirm address: {url}",
    "Click to activate your reward points: {url}",
    "Verify your identity to continue using services: {url}",
]

PHISHING_URLS = [
    "http://mci-secure-verify.xyz/{token}",
    "https://account-confirm.net/login?id={token}",
    "http://bit.ly/{token}",
    "http://tinyurl.com/{token}",
    "https://prize-claim.info/{token}",
    "http://refund-portal.ru/{token}",
    "http://bank-secure-update.tk/{token}",
    "https://delivery-track.cf/{token}",
    "http://goo.gl/{token}",
    "http://secure-login-now.ml/{token}",
    "https://customer-verify.ga/{token}",
    "http://win-prize.xyz/{token}",
]

SUSPICIOUS_SENDERS = [
    "INFO", "ALERT", "NOTIFY", "SECURE", "BANK", "MCI-VERIFY",
    "+989100000001", "+447911123456", "0000", "9999",
    "PRIZE", "WINNER", "FREE-GIFT",
]

# ── Normal SMS templates ──────────────────────────────────────────────────────
NORMAL_TEMPLATES = [
    "Your OTP code is {otp}. Valid for 5 minutes. Do not share.",
    "Hi {name}, your appointment is confirmed for {date} at {time}.",
    "Your order #{order_id} has been shipped. Expected delivery: {date}.",
    "Dear {name}, your monthly invoice is ready. Amount: {amount} IRR.",
    "MCI: Your balance is {amount} IRR. Dial *141# to recharge.",
    "Reminder: your subscription renews on {date}.",
    "Thank you for your payment of {amount} IRR. Receipt: #{order_id}.",
    "Your account statement for {month} is available in the app.",
    "Welcome back, {name}! You have {points} reward points.",
    "Your call to {phone} lasted {duration} minutes. Cost: {amount} IRR.",
    "Scheduled maintenance on {date} from 02:00 to 04:00. Sorry for inconvenience.",
    "Hi {name}, we received your support ticket #{order_id}.",
    "Your data package of {amount} MB is running low.",
    "New message from {name} in your inbox.",
    "Your bill payment of {amount} IRR was successful.",
    "Delivery confirmed for order #{order_id}. Thank you!",
    "Your verification code: {otp}. Expires in 10 minutes.",
    "MCI: International roaming activated for your number.",
    "Congratulations on completing {month} with us! Loyalty reward added.",
    "Your package has arrived at the post office. ID: #{order_id}.",
]

NORMAL_SENDERS = [
    "MCI", "Hamrah-Aval", "Bank-Melli", "Post", "IRANCELL",
    "Snapp", "Digikala", "Tapsi", "ZarinPal", "ShopApp",
    "+98912345678", "+98911234567", "+98913456789",
]

# ── Helper generators ─────────────────────────────────────────────────────────

def random_token(length=8):
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choices(chars, k=length))

def random_phone():
    return f"+9891{random.randint(10000000, 99999999)}"

def random_date(days_back=90):
    base = datetime.now() - timedelta(days=random.randint(0, days_back))
    return base.strftime("%Y-%m-%d %H:%M:%S")

def random_name():
    names = ["Ali", "Sara", "Mohammad", "Fateme", "Reza", "Maryam",
             "Hassan", "Zahra", "Ahmad", "Leila", "Mehdi", "Nasrin"]
    return random.choice(names)

def fill_phishing(template):
    return template.format(
        url=random.choice(PHISHING_URLS).format(token=random_token()),
        amount=random.randint(500_000, 50_000_000),
        discount=random.randint(30, 90),
        token=random_token(),
    )

def fill_normal(template):
    return template.format(
        otp=random.randint(100000, 999999),
        name=random_name(),
        date=(datetime.now() + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
        time=f"{random.randint(8,20):02d}:{random.choice(['00','15','30','45'])}",
        order_id=random.randint(100000, 999999),
        amount=random.randint(10_000, 5_000_000),
        month=datetime.now().strftime("%B"),
        points=random.randint(100, 5000),
        phone=random_phone(),
        duration=random.randint(1, 60),
    )

# ── Feature extraction helpers ────────────────────────────────────────────────

def has_url(text):
    return int(any(prefix in text.lower() for prefix in ["http://", "https://", "bit.ly", "tinyurl"]))

def has_suspicious_tld(text):
    suspicious_tlds = [".xyz", ".tk", ".ml", ".ga", ".cf", ".ru", ".info", ".net"]
    return int(any(tld in text.lower() for tld in suspicious_tlds))

def has_urgency_words(text):
    urgency = ["urgent", "immediately", "now", "expires", "suspended",
               "locked", "blocked", "final", "warning", "alert", "action required"]
    return int(any(word in text.lower() for word in urgency))

def has_prize_words(text):
    prize = ["won", "winner", "prize", "congratulations", "free", "gift",
             "lottery", "selected", "reward", "claim"]
    return int(any(word in text.lower() for word in prize))

def url_shortener_present(text):
    shorteners = ["bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly"]
    return int(any(s in text.lower() for s in shorteners))

def count_special_chars(text):
    return sum(1 for c in text if c in "!@#$%^&*()_+=[]{}|;:,.<>?")

def message_length(text):
    return len(text)

def digit_ratio(text):
    if not text:
        return 0.0
    digits = sum(1 for c in text if c.isdigit())
    return round(digits / len(text), 4)

# ── Main generator ────────────────────────────────────────────────────────────

def generate_dataset(n_samples=10000, phishing_ratio=0.3):
    """
    Generate a labeled SMS dataset.

    Args:
        n_samples: total number of messages
        phishing_ratio: fraction that are phishing (default 30%)

    Returns:
        pd.DataFrame with columns:
            timestamp, sender, message, label,
            has_url, has_suspicious_tld, has_urgency_words,
            has_prize_words, url_shortener_present,
            special_char_count, message_length, digit_ratio
    """
    n_phishing = int(n_samples * phishing_ratio)
    n_normal = n_samples - n_phishing

    records = []

    # Generate phishing messages
    for _ in range(n_phishing):
        template = random.choice(PHISHING_TEMPLATES)
        msg = fill_phishing(template)
        sender = random.choice(SUSPICIOUS_SENDERS)
        records.append({
            "timestamp": random_date(),
            "sender": sender,
            "message": msg,
            "label": 1,  # 1 = phishing
        })

    # Generate normal messages
    for _ in range(n_normal):
        template = random.choice(NORMAL_TEMPLATES)
        msg = fill_normal(template)
        sender = random.choice(NORMAL_SENDERS)
        records.append({
            "timestamp": random_date(),
            "sender": sender,
            "message": msg,
            "label": 0,  # 0 = normal
        })

    df = pd.DataFrame(records)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Extract features
    df["has_url"] = df["message"].apply(has_url)
    df["has_suspicious_tld"] = df["message"].apply(has_suspicious_tld)
    df["has_urgency_words"] = df["message"].apply(has_urgency_words)
    df["has_prize_words"] = df["message"].apply(has_prize_words)
    df["url_shortener_present"] = df["message"].apply(url_shortener_present)
    df["special_char_count"] = df["message"].apply(count_special_chars)
    df["message_length"] = df["message"].apply(message_length)
    df["digit_ratio"] = df["message"].apply(digit_ratio)

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate SMS phishing dataset")
    parser.add_argument("--output", default="data/sms_dataset.csv", help="Output CSV path")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--phishing-ratio", type=float, default=0.3, help="Fraction of phishing messages")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.samples} SMS messages ({args.phishing_ratio*100:.0f}% phishing)...")
    df = generate_dataset(n_samples=args.samples, phishing_ratio=args.phishing_ratio)

    df.to_csv(args.output, index=False)

    print(f"\nDataset saved to: {args.output}")
    print(f"Total messages  : {len(df)}")
    print(f"Phishing (1)    : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"Normal   (0)    : {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.1f}%)")
    print(f"\nSample rows:")
    print(df[["timestamp", "sender", "message", "label"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
