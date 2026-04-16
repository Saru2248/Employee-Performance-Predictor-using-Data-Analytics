"""
=============================================================
  Employee Performance Predictor
  Module: data_generator.py
  Purpose: Generate a realistic synthetic HR dataset
=============================================================

Why synthetic data?
  - We don't have access to real company HR records.
  - Synthetic data lets us simulate a real company environment.
  - It follows realistic distributions seen in industry datasets.
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── Constants ────────────────────────────────────────────────
N_EMPLOYEES = 1000          # total records to generate

DEPARTMENTS  = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "Support"]
GENDERS      = ["Male", "Female"]
EDUCATION    = ["Bachelor's", "Master's", "PhD", "Diploma"]
JOB_LEVELS   = ["Junior", "Mid", "Senior", "Lead", "Manager"]


def generate_employee_dataset(n: int = N_EMPLOYEES, save_path: str = None) -> pd.DataFrame:
    """
    Generate a synthetic HR employee dataset with realistic correlations.

    Parameters
    ----------
    n        : number of employee records
    save_path: if given, saves the CSV to that path

    Returns
    -------
    df : pandas DataFrame
    """

    # ── 1. Basic Identifiers ─────────────────────────────────
    employee_ids = [f"EMP{str(i).zfill(4)}" for i in range(1, n + 1)]

    # ── 2. Demographics ──────────────────────────────────────
    age = rng.integers(22, 60, size=n)
    gender = rng.choice(GENDERS, size=n)
    education = rng.choice(EDUCATION, size=n, p=[0.50, 0.30, 0.10, 0.10])

    # ── 3. Job Details ───────────────────────────────────────
    department = rng.choice(DEPARTMENTS, size=n)
    job_level  = rng.choice(JOB_LEVELS,  size=n, p=[0.30, 0.30, 0.20, 0.12, 0.08])

    # Experience: realistic — older employees usually have more
    experience_years = np.clip(
        (age - 22) * rng.uniform(0.4, 0.9, size=n) + rng.normal(0, 1, size=n),
        0, 35
    ).astype(int)

    # ── 4. Financial ─────────────────────────────────────────
    # Base salary influenced by job level
    level_salary_map = {"Junior": 35000, "Mid": 55000, "Senior": 80000,
                        "Lead": 100000, "Manager": 130000}
    base_salary = np.array([level_salary_map[l] for l in job_level])
    salary = (
        base_salary
        + rng.normal(0, 8000, size=n)
        + experience_years * 1200
    ).clip(25000, 200000).astype(int)

    # ── 5. Performance Drivers ───────────────────────────────
    training_hours        = rng.integers(10, 100, size=n)
    projects_completed    = rng.integers(1, 20,  size=n)
    monthly_working_hours = rng.integers(140, 220, size=n)
    overtime_hours        = rng.integers(0, 40, size=n)
    leaves_taken          = rng.integers(0, 20, size=n)

    # Manager rating (1–5, reflects how manager perceives the employee)
    manager_rating = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.10, 0.25, 0.40, 0.20])

    # Peer review score (0–10)
    peer_review_score = np.clip(rng.normal(6.5, 1.5, size=n), 0, 10).round(1)

    # Attendance rate (%)
    attendance_rate = np.clip(rng.normal(90, 8, size=n), 60, 100).round(1)

    # Years at company
    years_at_company = np.clip(experience_years - rng.integers(0, 5, size=n), 0, 35)

    # ── 6. Target Variable: Performance Score (0–100) ────────
    # Realistic weighted formula based on HR domain knowledge:
    performance_score = (
        0.25 * (manager_rating / 5 * 100)          # 25% manager rating
        + 0.20 * peer_review_score * 10             # 20% peer review
        + 0.15 * (attendance_rate)                  # 15% attendance
        + 0.15 * (training_hours / 100 * 100)       # 15% training
        + 0.15 * (projects_completed / 20 * 100)    # 15% projects
        + 0.10 * (1 - overtime_hours / 40) * 100    # 10% work-life balance
        + rng.normal(0, 3, size=n)                   # small noise
    ).clip(0, 100).round(2)

    # ── 7. Performance Category (Target Label) ───────────────
    # High (score >= 75), Medium (50–74), Low (< 50)
    performance_category = pd.cut(
        performance_score,
        bins=[-1, 50, 74, 101],
        labels=["Low", "Medium", "High"]
    )

    # ── 8. Additional HR Fields ──────────────────────────────
    promotion_last_5_years = rng.choice([0, 1], size=n, p=[0.70, 0.30])
    # High performers more likely promoted
    promotion_last_5_years = np.where(performance_score >= 75,
                                       rng.choice([0, 1], size=n, p=[0.30, 0.70]),
                                       promotion_last_5_years)

    # Attrition risk
    attrition_risk = np.where(performance_score < 50, "High",
                     np.where(performance_score < 75, "Medium", "Low"))

    # ── 9. Build DataFrame ───────────────────────────────────
    df = pd.DataFrame({
        "employee_id":           employee_ids,
        "age":                   age,
        "gender":                gender,
        "education":             education,
        "department":            department,
        "job_level":             job_level,
        "experience_years":      experience_years,
        "salary":                salary,
        "training_hours":        training_hours,
        "projects_completed":    projects_completed,
        "monthly_working_hours": monthly_working_hours,
        "overtime_hours":        overtime_hours,
        "leaves_taken":          leaves_taken,
        "manager_rating":        manager_rating,
        "peer_review_score":     peer_review_score,
        "attendance_rate":       attendance_rate,
        "years_at_company":      years_at_company,
        "promotion_last_5_years":promotion_last_5_years,
        "attrition_risk":        attrition_risk,
        "performance_score":     performance_score,
        "performance_category":  performance_category,
    })

    # ── 10. Save ─────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[✓] Dataset saved → {save_path}")

    print(f"[✓] Generated {n} employee records.")
    print(f"    Performance breakdown:\n{df['performance_category'].value_counts().to_string()}")

    return df


# =============================================================
# Quick test – run directly
# =============================================================
if __name__ == "__main__":
    df = generate_employee_dataset(
        n=1000,
        save_path="data/raw/employee_data.csv"
    )
    print(df.head())
    print(df.dtypes)
