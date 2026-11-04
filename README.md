# FinGuard AI: Real-Time FinTech Fraud Detection with Explainable AI (XAI)

An end-to-end production machine learning system that detects fraudulent financial transactions in real time and explains the exact risk drivers behind every prediction using **SHAP (SHapley Additive exPlanations)**.

![Dashboard Preview](dashboard_preview.png)

## Key Engineering Highlights
- **Imbalanced Learning Pipeline:** Handled severe class imbalance (~1.5% fraud rate) using cost-sensitive `XGBoost` (`scale_pos_weight`) and Precision-Recall threshold tuning.
- **Domain Feature Engineering:** Built behavioral velocity and deviation features (`amount_to_avg_ratio`, `is_late_night`, `txn_count_last_1hr`, `distance_from_home_km`).
- **Real-Time SHAP Explainability:** Integrated `shap.TreeExplainer` to compute log-odds feature attributions for every transaction in under 25ms.
- **Low-Latency FastAPI + Interactive UI:** Served predictions via FastAPI (`Uvicorn`) connected to a responsive Tailwind CSS & Chart.js dark-mode monitoring dashboard.

## Tech Stack
- **Machine Learning:** Python, Scikit-Learn, XGBoost, SHAP, Pandas, NumPy
- **Backend & MLOps:** FastAPI, Uvicorn, Pydantic, Joblib, Docker
- **Frontend Dashboard:** HTML5, Tailwind CSS, Chart.js

## How to Run Locally
```bash
pip install -r requirements.txt
python src/train_pipeline.py
uvicorn api.app:app --reload