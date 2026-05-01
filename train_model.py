"""
train_model.py
--------------
Loads the SMS dataset, trains a Random Forest classifier,
evaluates it, and saves the trained model to disk.

Usage:
    python train_model.py
    python train_model.py --data data/sms_dataset.csv --model models/phishing_model.pkl
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Feature columns used for training ────────────────────────────────────────
FEATURE_COLS = [
    "has_url",
    "has_suspicious_tld",
    "has_urgency_words",
    "has_prize_words",
    "url_shortener_present",
    "special_char_count",
    "message_length",
    "digit_ratio",
]

LABEL_COL = "label"


# ── Load & validate data ──────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    print(f"Loaded {len(df)} rows — {df[LABEL_COL].sum()} phishing, "
          f"{(df[LABEL_COL] == 0).sum()} normal")
    return df


# ── Train ─────────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, model_type: str = "random_forest"):
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain size : {len(X_train)}")
    print(f"Test size  : {len(X_test)}")

    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",  # handles imbalanced classes
            random_state=42,
            n_jobs=-1,
        )
    else:
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=42,
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", classifier),
    ])

    print(f"\nTraining {model_type} model...")
    pipeline.fit(X_train, y_train)

    return pipeline, X_train, X_test, y_train, y_test


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(pipeline, X_train, X_test, y_train, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Phishing"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix:")
    print(f"  True Negatives  (correctly flagged normal)  : {tn}")
    print(f"  False Positives (normal flagged as phishing): {fp}")
    print(f"  False Negatives (phishing missed)           : {fn}")
    print(f"  True Positives  (phishing correctly caught) : {tp}")

    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\nROC-AUC Score : {roc_auc:.4f}")
    print(f"F1 Score      : {f1:.4f}")

    # Cross-validation on training set
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
    print(f"\n5-Fold CV F1  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance (Random Forest only)
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        print("\nFeature Importances:")
        importances = sorted(
            zip(FEATURE_COLS, clf.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        for feat, score in importances:
            bar = "█" * int(score * 40)
            print(f"  {feat:<30} {score:.4f}  {bar}")

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "f1_score": round(f1, 4),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
    }

    return metrics


# ── Save model ────────────────────────────────────────────────────────────────

def save_model(pipeline, metrics, model_path: str):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    metrics_path = Path(model_path).with_suffix(".json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved   : {model_path}")
    print(f"Metrics saved : {metrics_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SMS phishing detection model")
    parser.add_argument("--data", default="data/sms_dataset.csv")
    parser.add_argument("--model", default="models/phishing_model.pkl")
    parser.add_argument("--model-type", default="random_forest",
                        choices=["random_forest", "logistic_regression"])
    args = parser.parse_args()

    df = load_data(args.data)
    pipeline, X_train, X_test, y_train, y_test = train(df, args.model_type)
    metrics = evaluate(pipeline, X_train, X_test, y_train, y_test)
    save_model(pipeline, metrics, args.model)

    print("\nDone! Next step: run detector.py to score new messages.")


if __name__ == "__main__":
    main()
