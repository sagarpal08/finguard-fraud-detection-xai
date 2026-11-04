import time
import shap
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="FinTech Fraud Detection & XAI API")

# Enable CORS for React.js Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model & initialize SHAP TreeExplainer at startup
model = joblib.load("models/xgb_fraud_model.joblib")
feature_cols = joblib.load("models/feature_columns.joblib")
explainer = shap.TreeExplainer(model)

class TransactionInput(BaseModel):
    txn_amount: float
    user_avg_amount_30d: float
    distance_from_home_km: float
    txn_count_last_1hr: int
    hour_of_day: int
    is_international: int
    failed_pin_attempts: int

@app.post("/predict")
def predict_fraud(txn: TransactionInput):
    start_time = time.time()

    # 1. Real-time Feature Engineering
    amount_to_avg_ratio = round(txn.txn_amount / max(txn.user_avg_amount_30d, 1.0), 2)
    is_late_night = 1 if (1 <= txn.hour_of_day <= 5) else 0

    input_data = pd.DataFrame([{
        "txn_amount": txn.txn_amount,
        "user_avg_amount_30d": txn.user_avg_amount_30d,
        "amount_to_avg_ratio": amount_to_avg_ratio,
        "distance_from_home_km": txn.distance_from_home_km,
        "txn_count_last_1hr": txn.txn_count_last_1hr,
        "hour_of_day": txn.hour_of_day,
        "is_late_night": is_late_night,
        "is_international": txn.is_international,
        "failed_pin_attempts": txn.failed_pin_attempts
    }])[feature_cols]

    # 2. Model Prediction
    fraud_prob = float(model.predict_proba(input_data)[0][1])
    risk_level = "HIGH RISK (BLOCKED)" if fraud_prob >= 0.75 else ("MEDIUM RISK (OTP REQUIRED)" if fraud_prob >= 0.40 else "LOW RISK (APPROVED)")

    # 3. SHAP Explainability (Top 3 Reasons)
    shap_values = explainer.shap_values(input_data)
    feature_impacts = []
    for col, val, shap_val in zip(feature_cols, input_data.iloc[0], shap_values[0]):
        feature_impacts.append({
            "feature": col,
            "value": float(val),
            "shap_impact": round(float(shap_val), 4)
        })

    # Sort by highest positive impact pushing towards fraud
    top_reasons = sorted(feature_impacts, key=lambda x: abs(x["shap_impact"]), reverse=True)[:4]
    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "fraud_probability": round(fraud_prob * 100, 2),
        "risk_status": risk_level,
        "latency_ms": latency_ms,
        "engineered_features": {
            "amount_to_avg_ratio": amount_to_avg_ratio,
            "is_late_night": bool(is_late_night)
        },
        "xai_top_drivers": top_reasons
    }