"""
=============================================================
  Employee Performance Predictor
  File: main.py  ←  ENTRY POINT
  Purpose: Orchestrate the complete pipeline end-to-end
=============================================================

Run this file to:
  1. Generate synthetic HR dataset
  2. Preprocess & engineer features
  3. Run full EDA (saves 8 graphs)
  4. Train 5 ML models & compare
  5. Evaluate best model
  6. Save model + artifacts
  7. Run demo predictions

Usage:
  python main.py
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Ensure project root is on Python path ────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Imports ───────────────────────────────────────────────────
from src.data_generator import generate_employee_dataset
from src.preprocessing  import full_pipeline
from src.eda            import run_full_eda
from src.model_trainer  import (
    train_all_models, evaluate_model,
    plot_feature_importance, save_model
)
from src.predictor      import predict_single_employee, predict_batch

import pandas as pd


# =============================================================
# Paths
# =============================================================
RAW_DATA_PATH       = "data/raw/employee_data.csv"
PROCESSED_DATA_PATH = "data/processed/employee_processed.csv"


# =============================================================
def main():
    print("=" * 60)
    print("  [HR] EMPLOYEE PERFORMANCE PREDICTOR")
    print("     Industry-Oriented Data Science Project")
    print("=" * 60)

    # ── PHASE 1: Data Generation ─────────────────────────────
    print("\n[PHASE 1] Generating Synthetic HR Dataset...")
    df_raw = generate_employee_dataset(n=1000, save_path=RAW_DATA_PATH)

    # ── PHASE 2: EDA ─────────────────────────────────────────
    print("\n[PHASE 2] Running Exploratory Data Analysis...")
    run_full_eda(df_raw)

    # ── PHASE 3: Preprocessing ───────────────────────────────
    print("\n[PHASE 3] Preprocessing & Feature Engineering...")
    X, y, feature_names, class_names, df_processed = full_pipeline(
        raw_path=RAW_DATA_PATH,
        processed_path=PROCESSED_DATA_PATH
    )

    # ── PHASE 4: Model Training ───────────────────────────────
    print("\n[PHASE 4] Training & Comparing ML Models...")
    best_model, best_name, results_df, X_train, X_test, y_train, y_test = \
        train_all_models(X, y, feature_names, class_names)

    # ── PHASE 5: Evaluation ───────────────────────────────────
    print("\n[PHASE 5] Evaluating Best Model...")
    metrics = evaluate_model(best_model, X_test, y_test, class_names, best_name)

    # ── PHASE 6: Feature Importance ───────────────────────────
    print("\n[PHASE 6] Plotting Feature Importances...")
    plot_feature_importance(best_model, feature_names, best_name)

    # ── PHASE 7: Save Model ───────────────────────────────────
    print("\n[PHASE 7] Saving Best Model...")
    save_model(best_model, best_name)

    # ── PHASE 8: Demo Prediction ──────────────────────────────
    print("\n[PHASE 8] Demo: Predicting for Sample Employees...")

    # Sample 1: High-potential employee
    emp1 = {
        "age": 35, "gender": "Female", "education": "Master's",
        "department": "Engineering", "job_level": "Senior",
        "experience_years": 10, "salary": 95000,
        "training_hours": 80, "projects_completed": 15,
        "monthly_working_hours": 175, "overtime_hours": 8,
        "leaves_taken": 4, "manager_rating": 5,
        "peer_review_score": 9.0, "attendance_rate": 97.5,
        "years_at_company": 7, "promotion_last_5_years": 1,
    }

    # Sample 2: Struggling employee
    emp2 = {
        "age": 28, "gender": "Male", "education": "Diploma",
        "department": "Support", "job_level": "Junior",
        "experience_years": 2, "salary": 35000,
        "training_hours": 15, "projects_completed": 3,
        "monthly_working_hours": 195, "overtime_hours": 35,
        "leaves_taken": 18, "manager_rating": 2,
        "peer_review_score": 4.0, "attendance_rate": 72.0,
        "years_at_company": 1, "promotion_last_5_years": 0,
    }

    for i, emp in enumerate([emp1, emp2], 1):
        print(f"\n  ── Employee {i} ────────────────────────────────────")
        result = predict_single_employee(emp, model=best_model)
        print(f"    Predicted Category  : {result['predicted_category']}")
        print(f"    Confidence Scores   : {result['confidence_%']}")
        print(f"    HR Recommendation   : {result['recommendation']}")
        print(f"    Attrition Risk      : {result['attrition_risk']}")

    # ── PHASE 9: Batch Prediction ─────────────────────────────
    print("\n[PHASE 9] Batch Prediction on Processed Dataset...")
    df_batch = pd.read_csv(RAW_DATA_PATH).head(50)
    batch_results = predict_batch(df_batch, model=best_model)
    print(batch_results.head(10).to_string(index=False))

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [OK] PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n  Best Model   : {best_name}")
    print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score     : {metrics['f1']:.4f}")
    if metrics['roc_auc']:
        print(f"  ROC-AUC      : {metrics['roc_auc']:.4f}")
    print("\n  Outputs:")
    print("    [>] data/raw/employee_data.csv       <- raw dataset")
    print("    [>] data/processed/               <- cleaned + features")
    print("    [>] images/                       <- all plots")
    print("    [>] models/                       <- saved model & encoders")
    print("    [>] outputs/                      <- reports & predictions")
    print("\n  Next step: run  ->  streamlit run dashboard.py")
    print("=" * 60)


# =============================================================
if __name__ == "__main__":
    main()
