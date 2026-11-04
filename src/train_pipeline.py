import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# 1. Generate Realistic Imbalanced FinTech Transaction Data (50,000 rows)
def load_or_create_data():
    np.random.seed(42)
    n_samples = 50000
    
    df = pd.DataFrame({
        "txn_amount": np.random.exponential(scale=120, size=n_samples).round(2),
        "user_avg_amount_30d": np.random.uniform(30, 200, size=n_samples).round(2),
        "distance_from_home_km": np.random.exponential(scale=15, size=n_samples).round(2),
        "txn_count_last_1hr": np.random.poisson(lam=1.2, size=n_samples),
        "hour_of_day": np.random.randint(0, 24, size=n_samples),
        "is_international": np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        "failed_pin_attempts": np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.92, 0.05, 0.02, 0.01])
    })

    # Feature Engineering (Interview Highlight!)
    df["amount_to_avg_ratio"] = (df["txn_amount"] / df["user_avg_amount_30d"]).round(2)
    df["is_late_night"] = df["hour_of_day"].apply(lambda h: 1 if (h >= 1 and h <= 5) else 0)

    # Realistic Fraud Logic (~1.5% - 2% Fraud Rate)
    fraud_score = (
        (df["amount_to_avg_ratio"] > 4.0).astype(int) * 2.5 +
        (df["distance_from_home_km"] > 80).astype(int) * 2.0 +
        (df["txn_count_last_1hr"] >= 4).astype(int) * 2.0 +
        (df["is_late_night"] == 1).astype(int) * 1.5 +
        (df["is_international"] == 1).astype(int) * 2.0 +
        (df["failed_pin_attempts"] >= 2).astype(int) * 2.5 +
        np.random.normal(0, 1.2, size=n_samples)
    )
    df["is_fraud"] = (fraud_score > 5.8).astype(int)
    return df

def train_model():
    df = load_or_create_data()
    print(f"Total Transactions: {len(df)} | Fraud Cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")

    feature_cols = [
        "txn_amount", "user_avg_amount_30d", "amount_to_avg_ratio",
        "distance_from_home_km", "txn_count_last_1hr", "hour_of_day",
        "is_late_night", "is_international", "failed_pin_attempts"
    ]
    X = df[feature_cols]
    y = df["is_fraud"]

    # Stratified Split taaki train aur test dono mein fraud ratio barabar rahe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Imbalance Ratio calculate karna (Negative / Positive)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # XGBoost Classifier with Cost-Sensitive Learning
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions & Evaluation
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.5).astype(int)

    print("\n--- Model Evaluation Metrics ---")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
    print(f"PR-AUC Score (Precision-Recall): {average_precision_score(y_test, y_probs):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save Model & Feature Names
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_fraud_model.joblib")
    joblib.dump(feature_cols, "models/feature_columns.joblib")
    print("\nModel successfully saved in 'models/' directory!")

if __name__ == "__main__":
    train_model()