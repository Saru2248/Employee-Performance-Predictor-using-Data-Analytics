"""
=============================================================
  Employee Performance Predictor
  Module: predictor.py
  Purpose: Load saved model and predict new employee performance
=============================================================

This is the INFERENCE module — used after the model is trained.
HR managers would use this to score new employees.
"""

import pandas as pd
import numpy as np
import joblib
import os


MODEL_DIR  = "models"
LABEL_ENC  = os.path.join(MODEL_DIR, "label_encoders.pkl")
SCALER_PKL = os.path.join(MODEL_DIR, "scaler.pkl")

# Map model names to file names
MODEL_MAP = {
    "random_forest":     "random_forest_model.pkl",
    "xgboost":           "xgboost_model.pkl",
    "gradient_boosting": "gradient_boosting_model.pkl",
    "logistic_regression": "logistic_regression_model.pkl",
    "svm":               "svm_model.pkl",
}


def load_best_model(model_key: str = None):
    """
    Load the saved best model.
    If model_key is None, auto-detect from models/ directory.
    """
    if model_key and model_key in MODEL_MAP:
        path = os.path.join(MODEL_DIR, MODEL_MAP[model_key])
    else:
        # Auto-detect: pick first .pkl that isn't encoder/scaler
        pkls = [f for f in os.listdir(MODEL_DIR)
                if f.endswith(".pkl") and "encoder" not in f and "scaler" not in f]
        if not pkls:
            raise FileNotFoundError("No trained model found in models/ folder. Run main.py first.")
        path = os.path.join(MODEL_DIR, sorted(pkls)[0])

    model = joblib.load(path)
    print(f"[✓] Loaded model from: {path}")
    return model


def predict_single_employee(employee_dict: dict, model=None) -> dict:
    """
    Predict performance for ONE employee given as a dictionary.

    Parameters
    ----------
    employee_dict : raw employee attributes (same columns as training data)
    model         : pre-loaded model (if None, loads automatically)

    Returns
    -------
    result dict with predicted category and confidence scores
    """
    if model is None:
        model = load_best_model()

    encoders = joblib.load(LABEL_ENC)
    scaler   = joblib.load(SCALER_PKL)

    df = pd.DataFrame([employee_dict])

    # ── Feature Engineering (same as training) ───────────────
    df["salary_per_exp"]      = df["salary"] / (df["experience_years"] + 1)
    df["projects_per_year"]   = df["projects_completed"] / (df["years_at_company"] + 1)
    df["training_efficiency"] = df["training_hours"] / (df["projects_completed"] + 1)
    df["overtime_ratio"]      = df["overtime_hours"] / (df["monthly_working_hours"] + 1)
    df["engagement_score"]    = (
        (df["attendance_rate"] / 100) * 0.4
        + (df["training_hours"] / 100) * 0.3
        + (df["peer_review_score"] / 10) * 0.3
    )
    df["is_senior"] = df["job_level"].isin(["Senior", "Lead", "Manager"]).astype(int)

    # ── Encode Categoricals ────────────────────────────────────
    cat_cols = ["gender", "education", "department", "job_level"]
    for col in cat_cols:
        if col in df.columns and col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))

    # ── Drop unnecessary columns ───────────────────────────────
    drop_cols = ["employee_id", "attrition_risk", "performance_score",
                 "performance_category"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Scale ─────────────────────────────────────────────────
    numeric_cols = [
        "age", "experience_years", "salary", "training_hours",
        "projects_completed", "monthly_working_hours", "overtime_hours",
        "leaves_taken", "manager_rating", "peer_review_score",
        "attendance_rate", "years_at_company", "promotion_last_5_years",
        "salary_per_exp", "projects_per_year",
        "training_efficiency", "overtime_ratio", "engagement_score"
    ]
    scale_cols = [c for c in numeric_cols if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])

    # ── Predict ────────────────────────────────────────────────
    pred_encoded = model.predict(df.values)[0]
    target_le    = encoders["target"]
    pred_label   = target_le.inverse_transform([pred_encoded])[0]

    proba = model.predict_proba(df.values)[0] if hasattr(model, "predict_proba") else None
    confidence_map = {}
    if proba is not None:
        for cls, prob in zip(target_le.classes_, proba):
            confidence_map[cls] = round(float(prob) * 100, 2)

    result = {
        "predicted_category": pred_label,
        "confidence_%":       confidence_map,
        "recommendation":     _get_recommendation(pred_label),
        "attrition_risk":     _get_attrition_risk(pred_label),
    }
    return result


def predict_batch(df_raw: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Predict performance for a batch of employees.
    Returns a copy of df_raw with added prediction columns.
    """
    if model is None:
        model = load_best_model()

    predictions = []
    for _, row in df_raw.iterrows():
        try:
            result = predict_single_employee(row.to_dict(), model)
            predictions.append({
                "employee_id":     row.get("employee_id", "N/A"),
                "predicted_perf":  result["predicted_category"],
                "confidence_%":    max(result["confidence_%"].values()) if result["confidence_%"] else None,
                "recommendation":  result["recommendation"],
                "attrition_risk":  result["attrition_risk"],
            })
        except Exception as e:
            predictions.append({
                "employee_id":    row.get("employee_id", "N/A"),
                "predicted_perf": "Error",
                "confidence_%":   None,
                "recommendation": str(e),
                "attrition_risk": "Unknown",
            })

    pred_df = pd.DataFrame(predictions)
    print(f"[✓] Batch prediction complete: {len(pred_df)} records")
    pred_df.to_csv("outputs/batch_predictions.csv", index=False)
    print("[✓] Saved → outputs/batch_predictions.csv")
    return pred_df


# ─────────────────────────────────────────────────────────────
# HR Recommendation Engine
# ─────────────────────────────────────────────────────────────
def _get_recommendation(category: str) -> str:
    recs = {
        "High":   "⭐ High Performer — Consider for promotion, leadership roles, or mentoring programs.",
        "Medium": "📈 Moderate Performer — Provide targeted training, set SMART goals, increase engagement.",
        "Low":    "⚠️ Low Performer — Immediate intervention needed: PIP (Performance Improvement Plan), coaching.",
    }
    return recs.get(category, "No recommendation available.")


def _get_attrition_risk(category: str) -> str:
    risks = {
        "High":   "Low — Strong performer likely satisfied",
        "Medium": "Medium — Monitor carefully",
        "Low":    "High — At risk of leaving or disengagement",
    }
    return risks.get(category, "Unknown")


# =============================================================
# Demo Prediction
# =============================================================
if __name__ == "__main__":
    # Example: Predict performance for one hypothetical employee
    sample_employee = {
        "age": 32,
        "gender": "Female",
        "education": "Master's",
        "department": "Engineering",
        "job_level": "Mid",
        "experience_years": 8,
        "salary": 70000,
        "training_hours": 65,
        "projects_completed": 12,
        "monthly_working_hours": 180,
        "overtime_hours": 10,
        "leaves_taken": 5,
        "manager_rating": 4,
        "peer_review_score": 7.8,
        "attendance_rate": 93.5,
        "years_at_company": 5,
        "promotion_last_5_years": 1,
    }

    print("\n── Predicting Performance for Sample Employee ────────")
    result = predict_single_employee(sample_employee)
    print(f"\n  Predicted Category : {result['predicted_category']}")
    print(f"  Confidence         : {result['confidence_%']}")
    print(f"  Recommendation     : {result['recommendation']}")
    print(f"  Attrition Risk     : {result['attrition_risk']}")
