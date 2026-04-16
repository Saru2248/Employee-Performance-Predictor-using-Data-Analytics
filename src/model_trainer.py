"""
=============================================================
  Employee Performance Predictor
  Module: model_trainer.py
  Purpose: Train, compare, evaluate, and save ML models
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, f1_score
)
from xgboost import XGBClassifier

# ── Style ─────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#1E1E2E",
    "axes.facecolor":   "#2A2A3E",
    "text.color":       "#CDD6F4",
    "axes.labelcolor":  "#CDD6F4",
    "xtick.color":      "#CDD6F4",
    "ytick.color":      "#CDD6F4",
})

MODEL_DIR  = "models"
IMAGE_DIR  = "images"
REPORT_DIR = "outputs"

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(IMAGE_DIR,  exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TEST_SIZE    = 0.20   # 80/20 train-test split
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────
# Model Zoo
# ─────────────────────────────────────────────────────────────
def get_models(class_names) -> dict:
    """Return a dict of model_name → sklearn estimator."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=4,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=RANDOM_STATE
        ),
    }


# ─────────────────────────────────────────────────────────────
# Train & Evaluate All Models
# ─────────────────────────────────────────────────────────────
def train_all_models(X, y, feature_names, class_names):
    """
    Train all classifiers, compare them, and return the best one.

    Returns
    -------
    best_model   : fitted sklearn estimator
    best_name    : model name string
    results_df   : DataFrame with accuracy scores for each model
    X_test, y_test: for later evaluation
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = get_models(class_names)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    print("\n── Model Training & Cross-Validation ────────────────")
    for name, model in models.items():
        # 5-fold cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        # Fit on full training set
        model.fit(X_train, y_train)
        # Test set accuracy
        test_acc = accuracy_score(y_test, model.predict(X_test))
        f1       = f1_score(y_test, model.predict(X_test), average="weighted")

        results[name] = {
            "CV Mean Acc": round(cv_scores.mean(), 4),
            "CV Std":      round(cv_scores.std(),  4),
            "Test Acc":    round(test_acc, 4),
            "F1 Score":    round(f1, 4),
        }
        print(f"  {name:<25} CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
              f"Test={test_acc:.4f}  F1={f1:.4f}")

    # ── Select best model (highest test accuracy) ────────────
    results_df = pd.DataFrame(results).T.sort_values("Test Acc", ascending=False)
    best_name  = results_df["Test Acc"].idxmax()
    best_model = models[best_name]
    best_model.fit(X_train, y_train)   # ensure fitted on training data

    print(f"\n  [★] Best Model: {best_name}  (Test Acc = {results_df.loc[best_name, 'Test Acc']})")

    # ── Save results ──────────────────────────────────────────
    results_df.to_csv(f"{REPORT_DIR}/model_comparison.csv")
    print(f"  [✓] Model comparison saved → {REPORT_DIR}/model_comparison.csv")

    # ── Plot comparison ───────────────────────────────────────
    _plot_model_comparison(results_df)

    return best_model, best_name, results_df, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────
# Evaluate Best Model
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, class_names, model_name="Best Model"):
    """
    Full evaluation: accuracy, report, confusion matrix, ROC-AUC.
    Saves confusion matrix plot.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    roc = roc_auc_score(y_test, y_prob, multi_class="ovr") if y_prob is not None else None

    print(f"\n── Evaluation: {model_name} ─────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    if roc: print(f"  ROC-AUC   : {roc:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Save report
    report_txt = classification_report(y_test, y_pred, target_names=class_names)
    with open(f"{REPORT_DIR}/classification_report.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n\n")
        f.write(report_txt)
    print(f"  [✓] Report saved → {REPORT_DIR}/classification_report.txt")

    # Confusion matrix plot
    _plot_confusion_matrix(y_test, y_pred, class_names, model_name)

    return {"accuracy": acc, "f1": f1, "roc_auc": roc}


# ─────────────────────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names, model_name="Best Model"):
    """Plot top 15 feature importances (for tree/ensemble models)."""
    if not hasattr(model, "feature_importances_"):
        print(f"  [!] {model_name} does not have feature_importances_. Skipping.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]   # top 15

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#1E1E2E")
    ax.set_facecolor("#2A2A3E")

    colors = sns.color_palette("viridis", 15)
    ax.barh(
        [feature_names[i] for i in indices][::-1],
        importances[indices][::-1],
        color=colors, edgecolor="#1E1E2E"
    )

    ax.set_title(f"Top 15 Feature Importances — {model_name}",
                 fontsize=13, color="#CDD6F4", pad=10)
    ax.set_xlabel("Importance", fontsize=11)

    fig.tight_layout()
    path = f"{IMAGE_DIR}/09_feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] Feature importance plot saved → {path}")


# ─────────────────────────────────────────────────────────────
# Save Best Model
# ─────────────────────────────────────────────────────────────
def save_model(model, model_name: str):
    safe_name = model_name.replace(" ", "_").lower()
    path      = f"{MODEL_DIR}/{safe_name}_model.pkl"
    joblib.dump(model, path)
    print(f"  [✓] Model saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────
def _plot_model_comparison(results_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#1E1E2E")
    ax.set_facecolor("#2A2A3E")

    x     = np.arange(len(results_df))
    width = 0.35
    bars1 = ax.bar(x - width/2, results_df["CV Mean Acc"], width,
                   label="CV Acc", color="#89B4FA", edgecolor="#1E1E2E", alpha=0.9)
    bars2 = ax.bar(x + width/2, results_df["Test Acc"],    width,
                   label="Test Acc", color="#A6E3A1", edgecolor="#1E1E2E", alpha=0.9)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", color="#CDD6F4", fontsize=9)

    ax.set_title("Model Comparison — CV vs Test Accuracy",
                 fontsize=14, color="#CDD6F4", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)

    fig.tight_layout()
    path = f"{IMAGE_DIR}/10_model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] Model comparison plot saved → {path}")


def _plot_confusion_matrix(y_test, y_pred, class_names, model_name):
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1E1E2E")
    ax.set_facecolor("#2A2A3E")

    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", linewidths=0.5, ax=ax,
                annot_kws={"size": 14, "color": "#1E1E2E"})

    ax.set_xlabel("Predicted", fontsize=12, color="#CDD6F4")
    ax.set_ylabel("Actual",    fontsize=12, color="#CDD6F4")
    ax.set_title(f"Confusion Matrix — {model_name}",
                 fontsize=13, color="#CDD6F4")

    fig.tight_layout()
    path = f"{IMAGE_DIR}/11_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [✓] Confusion matrix saved → {path}")


# =============================================================
if __name__ == "__main__":
    from src.data_generator import generate_employee_dataset
    from src.preprocessing  import full_pipeline

    generate_employee_dataset(n=1000, save_path="data/raw/employee_data.csv")
    X, y, features, classes, df = full_pipeline(
        "data/raw/employee_data.csv", "data/processed/employee_processed.csv"
    )
    best_model, best_name, results_df, X_train, X_test, y_train, y_test = \
        train_all_models(X, y, features, classes)
    evaluate_model(best_model, X_test, y_test, classes, best_name)
    plot_feature_importance(best_model, features, best_name)
    save_model(best_model, best_name)
