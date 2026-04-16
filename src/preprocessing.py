"""
=============================================================
  Employee Performance Predictor
  Module: preprocessing.py
  Purpose: Data cleaning and feature engineering pipeline
=============================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── Column Groups ─────────────────────────────────────────────
NUMERIC_COLS = [
    "age", "experience_years", "salary", "training_hours",
    "projects_completed", "monthly_working_hours", "overtime_hours",
    "leaves_taken", "manager_rating", "peer_review_score",
    "attendance_rate", "years_at_company", "promotion_last_5_years"
]

CATEGORICAL_COLS = ["gender", "education", "department", "job_level"]

# Columns to drop (not useful as features)
DROP_COLS = ["employee_id", "attrition_risk", "performance_score", "performance_category"]

TARGET_COL = "performance_category"


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    df = pd.read_csv(path)
    print(f"[✓] Loaded data: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def check_quality(df: pd.DataFrame) -> None:
    """Print data quality report."""
    print("\n── DATA QUALITY REPORT ──────────────────────────")
    print(f"  Shape        : {df.shape}")
    print(f"  Duplicates   : {df.duplicated().sum()}")
    print(f"  Missing vals :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"  Dtypes       :\n{df.dtypes.value_counts()}")
    print("─────────────────────────────────────────────────\n")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 – Remove duplicates
    Step 2 – Drop rows with nulls in critical columns
    Step 3 – Clip outliers using IQR method
    """
    original_len = len(df)

    # Remove duplicates
    df = df.drop_duplicates()

    # Drop rows with any nulls (synthetic data rarely has them, but good practice)
    df = df.dropna(subset=NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL])

    # Clip numeric outliers (IQR method) — keeps data realistic
    for col in NUMERIC_COLS:
        if col in df.columns:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(Q1, Q3)

    print(f"[✓] Cleaned data: {original_len} → {len(df)} rows")
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new meaningful features from existing columns.
    Feature engineering improves model accuracy.
    """
    df = df.copy()

    # 1. Salary per year of experience
    df["salary_per_exp"] = (df["salary"] / (df["experience_years"] + 1)).round(2)

    # 2. Projects per year
    df["projects_per_year"] = (df["projects_completed"] / (df["years_at_company"] + 1)).round(2)

    # 3. Training efficiency (training hours ÷ projects done)
    df["training_efficiency"] = (df["training_hours"] / (df["projects_completed"] + 1)).round(2)

    # 4. Overtime ratio
    df["overtime_ratio"] = (df["overtime_hours"] / (df["monthly_working_hours"] + 1)).round(4)

    # 5. Engagement score (composite)
    df["engagement_score"] = (
        (df["attendance_rate"] / 100) * 0.4
        + (df["training_hours"] / 100) * 0.3
        + (df["peer_review_score"] / 10) * 0.3
    ).round(4)

    # 6. Seniority flag
    df["is_senior"] = (df["job_level"].isin(["Senior", "Lead", "Manager"])).astype(int)

    print(f"[✓] Feature engineering: added 6 new features → {df.shape[1]} total columns")
    return df


def encode_and_scale(df: pd.DataFrame, fit: bool = True,
                     encoder_path: str = "models/label_encoders.pkl",
                     scaler_path: str = "models/scaler.pkl"):
    """
    Encode categorical variables and scale numeric features.

    Parameters
    ----------
    df           : cleaned + engineered DataFrame
    fit          : True → fit new encoders/scaler; False → use saved ones
    encoder_path : path to save/load LabelEncoders
    scaler_path  : path to save/load StandardScaler

    Returns
    -------
    X      : feature matrix (numpy array)
    y      : encoded target labels
    feature_names : list of feature column names
    """
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)

    df = df.copy()
    y_raw = df[TARGET_COL].astype(str)

    # Build feature set (drop irrelevant columns)
    feature_cols = [c for c in df.columns if c not in DROP_COLS + [TARGET_COL]]

    # ── Encode categoricals ──────────────────────────────────
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            if col in feature_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        joblib.dump(encoders, encoder_path)
        print(f"[✓] Encoders saved → {encoder_path}")
    else:
        encoders = joblib.load(encoder_path)
        for col in CATEGORICAL_COLS:
            if col in feature_cols:
                df[col] = encoders[col].transform(df[col].astype(str))

    # ── Encode target ────────────────────────────────────────
    target_le = LabelEncoder()
    if fit:
        y = target_le.fit_transform(y_raw)
        encoders["target"] = target_le
        joblib.dump(encoders, encoder_path)
    else:
        encoders = joblib.load(encoder_path)
        target_le = encoders["target"]
        y = target_le.transform(y_raw)

    # ── Scale numerics ───────────────────────────────────────
    all_numeric = NUMERIC_COLS + [
        "salary_per_exp", "projects_per_year",
        "training_efficiency", "overtime_ratio", "engagement_score"
    ]
    scale_cols = [c for c in all_numeric if c in feature_cols]

    if fit:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        joblib.dump(scaler, scaler_path)
        print(f"[✓] Scaler saved → {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        df[scale_cols] = scaler.transform(df[scale_cols])

    X = df[feature_cols].values
    print(f"[✓] Encoded & scaled: X shape = {X.shape}, y shape = {y.shape}")
    print(f"    Classes: {target_le.classes_}")

    return X, y, feature_cols, target_le.classes_


def full_pipeline(raw_path: str, processed_path: str = None):
    """
    Complete preprocessing pipeline.
    Returns X, y, feature_names, class_names, and clean DataFrame.
    """
    df = load_data(raw_path)
    check_quality(df)
    df = clean_data(df)
    df = engineer_features(df)

    if processed_path:
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"[✓] Processed data saved → {processed_path}")

    X, y, feature_names, class_names = encode_and_scale(df, fit=True)
    return X, y, feature_names, class_names, df


# =============================================================
if __name__ == "__main__":
    X, y, features, classes, df = full_pipeline(
        raw_path="data/raw/employee_data.csv",
        processed_path="data/processed/employee_processed.csv"
    )
    print(f"\n Features: {features}")
    print(f" Classes : {classes}")
