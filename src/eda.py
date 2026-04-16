"""
=============================================================
  Employee Performance Predictor
  Module: eda.py
  Purpose: Exploratory Data Analysis & Visualization
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Styling ──────────────────────────────────────────────────
PALETTE   = "viridis"
FIG_DIR   = "images"
DPI       = 150
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#1E1E2E",
    "axes.facecolor":   "#2A2A3E",
    "axes.edgecolor":   "#555577",
    "text.color":       "#CDD6F4",
    "axes.labelcolor":  "#CDD6F4",
    "xtick.color":      "#CDD6F4",
    "ytick.color":      "#CDD6F4",
    "grid.color":       "#3A3A5E",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})

os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [✓] Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 1. Performance Category Distribution
# ─────────────────────────────────────────────────────────────
def plot_performance_distribution(df: pd.DataFrame):
    """Bar chart showing Low / Medium / High employee counts."""
    counts = df["performance_category"].value_counts()
    colors = ["#F38BA8", "#FAB387", "#A6E3A1"]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#1E1E2E")
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="#888", width=0.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                str(val), ha="center", va="bottom", color="#CDD6F4", fontsize=12, fontweight="bold")

    ax.set_title("Employee Performance Category Distribution", fontsize=14, color="#CDD6F4", pad=12)
    ax.set_xlabel("Performance Category", fontsize=11)
    ax.set_ylabel("Number of Employees", fontsize=11)
    ax.set_facecolor("#2A2A3E")
    _save(fig, "01_performance_distribution")


# ─────────────────────────────────────────────────────────────
# 2. Performance Score Histogram
# ─────────────────────────────────────────────────────────────
def plot_score_histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#1E1E2E")
    ax.set_facecolor("#2A2A3E")

    ax.hist(df["performance_score"], bins=30, color="#89B4FA",
            edgecolor="#1E1E2E", alpha=0.85)
    ax.axvline(df["performance_score"].mean(), color="#F38BA8",
               linestyle="--", linewidth=2, label=f"Mean = {df['performance_score'].mean():.1f}")
    ax.axvline(df["performance_score"].median(), color="#A6E3A1",
               linestyle="--", linewidth=2, label=f"Median = {df['performance_score'].median():.1f}")

    ax.set_title("Distribution of Performance Scores", fontsize=14, color="#CDD6F4")
    ax.set_xlabel("Performance Score (0–100)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=10)
    _save(fig, "02_score_histogram")


# ─────────────────────────────────────────────────────────────
# 3. Department-wise Performance
# ─────────────────────────────────────────────────────────────
def plot_department_performance(df: pd.DataFrame):
    dept_avg = df.groupby("department")["performance_score"].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#1E1E2E")
    ax.set_facecolor("#2A2A3E")
    bars = ax.barh(dept_avg.index, dept_avg.values,
                   color=sns.color_palette("coolwarm", len(dept_avg)),
                   edgecolor="#444")

    for bar, val in zip(bars, dept_avg.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", color="#CDD6F4", fontsize=10)

    ax.set_title("Average Performance Score by Department", fontsize=14, color="#CDD6F4")
    ax.set_xlabel("Avg Performance Score", fontsize=11)
    _save(fig, "03_department_performance")


# ─────────────────────────────────────────────────────────────
# 4. Correlation Heatmap
# ─────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#1E1E2E")
    ax.set_facecolor("#1E1E2E")

    sns.heatmap(
        corr, annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5,
        linecolor="#1E1E2E", ax=ax,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, color="#CDD6F4", pad=10)
    fig.tight_layout()
    _save(fig, "04_correlation_heatmap")


# ─────────────────────────────────────────────────────────────
# 5. Scatter: Training Hours vs Performance Score
# ─────────────────────────────────────────────────────────────
def plot_training_vs_performance(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#1E1E2E")
    ax.set_facecolor("#2A2A3E")

    cat_colors = {"Low": "#F38BA8", "Medium": "#FAB387", "High": "#A6E3A1"}
    for cat, grp in df.groupby("performance_category", observed=True):
        ax.scatter(grp["training_hours"], grp["performance_score"],
                   c=cat_colors.get(str(cat), "gray"),
                   alpha=0.6, edgecolors="#222", s=35, label=str(cat))

    ax.set_title("Training Hours vs Performance Score", fontsize=14, color="#CDD6F4")
    ax.set_xlabel("Training Hours", fontsize=11)
    ax.set_ylabel("Performance Score", fontsize=11)
    ax.legend(title="Category", fontsize=10)
    _save(fig, "05_training_vs_performance")


# ─────────────────────────────────────────────────────────────
# 6. Box Plot: Manager Rating vs Performance
# ─────────────────────────────────────────────────────────────
def plot_manager_rating_boxplot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#1E1E2E")
    ax.set_facecolor("#2A2A3E")

    df_plot = df[["manager_rating", "performance_score"]].copy()
    df_plot["manager_rating"] = df_plot["manager_rating"].astype(int)

    grouped = [df_plot[df_plot["manager_rating"] == r]["performance_score"].values
               for r in sorted(df_plot["manager_rating"].unique())]

    bp = ax.boxplot(grouped, patch_artist=True,
                    labels=[f"Rating {r}" for r in sorted(df_plot["manager_rating"].unique())])

    colors = ["#F38BA8", "#FAB387", "#F9E2AF", "#A6E3A1", "#89B4FA"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_title("Manager Rating vs Performance Score", fontsize=14, color="#CDD6F4")
    ax.set_xlabel("Manager Rating", fontsize=11)
    ax.set_ylabel("Performance Score", fontsize=11)
    _save(fig, "06_manager_rating_boxplot")


# ─────────────────────────────────────────────────────────────
# 7. Attrition Risk Distribution
# ─────────────────────────────────────────────────────────────
def plot_attrition_risk(df: pd.DataFrame):
    risk_counts = df["attrition_risk"].value_counts()
    colors = ["#A6E3A1", "#FAB387", "#F38BA8"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1E1E2E")
    for ax in axes:
        ax.set_facecolor("#2A2A3E")

    # Pie chart
    axes[0].pie(risk_counts.values, labels=risk_counts.index,
                autopct="%1.1f%%", colors=colors,
                textprops={"color": "#CDD6F4"},
                wedgeprops={"edgecolor": "#1E1E2E", "linewidth": 2})
    axes[0].set_title("Attrition Risk Breakdown", color="#CDD6F4", fontsize=13)

    # Bar chart
    axes[1].bar(risk_counts.index, risk_counts.values, color=colors, edgecolor="#333")
    axes[1].set_title("Attrition Risk Counts", color="#CDD6F4", fontsize=13)
    axes[1].set_xlabel("Risk Level", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)

    fig.tight_layout()
    _save(fig, "07_attrition_risk")


# ─────────────────────────────────────────────────────────────
# 8. Top 10 Performers Table / Bar
# ─────────────────────────────────────────────────────────────
def plot_top_performers(df: pd.DataFrame):
    top = df.nlargest(10, "performance_score")[
        ["employee_id", "department", "job_level", "performance_score"]
    ].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#1E1E2E")
    ax.set_facecolor("#2A2A3E")

    bars = ax.barh(top["employee_id"], top["performance_score"],
                   color=sns.color_palette("Spectral", 10)[::-1], edgecolor="#333")

    for bar, dept, level in zip(bars, top["department"], top["job_level"]):
        ax.text(bar.get_width() - 1, bar.get_y() + bar.get_height()/2,
                f"  {dept} | {level}", va="center", ha="right",
                color="#1E1E2E", fontsize=9, fontweight="bold")

    ax.set_title("Top 10 Performing Employees", fontsize=14, color="#CDD6F4")
    ax.set_xlabel("Performance Score", fontsize=11)
    ax.set_xlim(0, 105)
    _save(fig, "08_top_performers")


# ─────────────────────────────────────────────────────────────
# Run All EDA
# ─────────────────────────────────────────────────────────────
def run_full_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Run all EDA steps and return a summary statistics DataFrame."""
    print("\n── Running EDA ──────────────────────────────────")
    print(df.describe().round(2).to_string())

    plot_performance_distribution(df)
    plot_score_histogram(df)
    plot_department_performance(df)
    plot_correlation_heatmap(df)
    plot_training_vs_performance(df)
    plot_manager_rating_boxplot(df)
    plot_attrition_risk(df)
    plot_top_performers(df)

    print("[✓] All EDA plots saved to images/")
    return df.describe().round(2)


# =============================================================
if __name__ == "__main__":
    df = pd.read_csv("data/raw/employee_data.csv")
    run_full_eda(df)
