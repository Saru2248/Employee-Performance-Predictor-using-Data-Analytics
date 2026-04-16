"""
=============================================================
  Employee Performance Predictor
  File: dashboard.py
  Purpose: Interactive Streamlit HR Insights Dashboard
=============================================================

Run with:
  streamlit run dashboard.py
"""

import os
import sys
import warnings

# Suppress known Plotly <-> pandas FutureWarning (internal to plotly/express/_core.py)
warnings.filterwarnings(
    "ignore",
    message="When grouping with a length-1 list-like",
    category=FutureWarning,
)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express       as px
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #89B4FA, #CBA6F7, #F5C2E7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 1rem 0 0.5rem;
  }
  .sub-title {
    text-align: center;
    color: #9399b2;
    font-size: 1rem;
    margin-bottom: 2rem;
  }
  .metric-card {
    background: linear-gradient(135deg, #1E1E2E, #2A2A3E);
    border: 1px solid #313244;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin: 0.3rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #89B4FA;
  }
  .metric-label {
    font-size: 0.85rem;
    color: #9399b2;
    margin-top: 0.3rem;
  }
  .pred-high   { color: #A6E3A1; font-weight: 700; font-size: 1.3rem; }
  .pred-medium { color: #FAB387; font-weight: 700; font-size: 1.3rem; }
  .pred-low    { color: #F38BA8; font-weight: 700; font-size: 1.3rem; }
  .rec-box {
    background: #2A2A3E;
    border-left: 4px solid #89B4FA;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.95rem;
  }
  div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E1E2E, #181825);
  }
  .stButton > button {
    background: linear-gradient(135deg, #89B4FA, #CBA6F7);
    color: #1E1E2E;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-size: 1rem;
    width: 100%;
    transition: all 0.3s ease;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(137,180,250,0.4);
  }
</style>
""", unsafe_allow_html=True)


# ── Load Data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = "data/raw/employee_data.csv"
    if not os.path.exists(path):
        st.error("Dataset not found. Please run `python main.py` first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    """Load the saved best model and encoders."""
    try:
        import joblib
        pkls = [f for f in os.listdir("models")
                if f.endswith(".pkl") and "encoder" not in f and "scaler" not in f]
        if not pkls:
            return None
        model = joblib.load(os.path.join("models", sorted(pkls)[0]))
        return model
    except Exception:
        return None


# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown("## 🏢 HR Control Panel")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["📊 Dashboard Overview", "🔍 Predict Employee", "📈 Analytics Deep Dive", "ℹ️ About Project"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**Dataset Stats**")
    df = load_data()
    st.metric("Total Employees", len(df))
    st.metric("Departments", df["department"].nunique())
    st.metric("Avg Performance Score", f"{df['performance_score'].mean():.1f}")


# =============================================================
# PAGE 1: DASHBOARD OVERVIEW
# =============================================================
if page == "📊 Dashboard Overview":
    st.markdown('<div class="main-title">🏢 Employee Performance Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-powered HR Analytics Dashboard — predict, understand, and retain talent</div>', unsafe_allow_html=True)

    df = load_data()

    # ── KPI Row ────────────────────────────────────────────────
    high   = (df["performance_category"] == "High").sum()
    medium = (df["performance_category"] == "Medium").sum()
    low    = (df["performance_category"] == "Low").sum()
    atrit  = (df["attrition_risk"] == "High").sum()

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in [
        (c1, high,   "⭐ High Performers",    "#A6E3A1"),
        (c2, medium, "📈 Medium Performers",  "#FAB387"),
        (c3, low,    "⚠️ Low Performers",     "#F38BA8"),
        (c4, atrit,  "🚨 High Attrition Risk","#CBA6F7"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{color}'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts Row 1 ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Performance Category Distribution")
        counts = df["performance_category"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        fig = px.bar(counts, x="Category", y="Count",
                     color="Category",
                     color_discrete_map={"High":"#A6E3A1","Medium":"#FAB387","Low":"#F38BA8"},
                     template="plotly_dark", text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="#1E1E2E", paper_bgcolor="#1E1E2E")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Performance Score Distribution")
        fig2 = px.histogram(df, x="performance_score", nbins=30,
                            color_discrete_sequence=["#89B4FA"],
                            template="plotly_dark", opacity=0.85)
        fig2.update_layout(plot_bgcolor="#1E1E2E", paper_bgcolor="#1E1E2E")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts Row 2 ───────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Department-wise Avg Performance")
        dept_avg = (df.groupby("department")["performance_score"]
                      .mean().reset_index()
                      .sort_values("performance_score"))
        dept_avg.columns = ["Department", "Avg Score"]
        fig3 = px.bar(dept_avg, x="Avg Score", y="Department", orientation="h",
                      color="Avg Score", color_continuous_scale="viridis",
                      template="plotly_dark")
        fig3.update_layout(plot_bgcolor="#1E1E2E", paper_bgcolor="#1E1E2E", coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Training Hours vs Performance")
        fig4 = px.scatter(df, x="training_hours", y="performance_score",
                          color="performance_category",
                          color_discrete_map={"High":"#A6E3A1","Medium":"#FAB387","Low":"#F38BA8"},
                          opacity=0.7, template="plotly_dark",
                          hover_data=["employee_id", "department", "job_level"])
        fig4.update_layout(plot_bgcolor="#1E1E2E", paper_bgcolor="#1E1E2E")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Top Performers Table ───────────────────────────────────
    st.subheader("🏆 Top 10 Performers")
    top10 = df.nlargest(10, "performance_score")[
        ["employee_id", "department", "job_level", "experience_years",
         "performance_score", "performance_category"]
    ].reset_index(drop=True)
    st.dataframe(top10.style.background_gradient(subset=["performance_score"], cmap="Greens"),
                 use_container_width=True)


# =============================================================
# PAGE 2: PREDICT EMPLOYEE
# =============================================================
elif page == "🔍 Predict Employee":
    st.markdown('<div class="main-title">🔍 Predict Employee Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter employee details to get an AI-powered performance prediction</div>', unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.warning("⚠️ No trained model found. Run `python main.py` first to train the model.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown("### 👤 Employee Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            age            = st.slider("Age", 22, 60, 30)
            gender         = st.selectbox("Gender", ["Male", "Female"])
            education      = st.selectbox("Education", ["Bachelor's", "Master's", "PhD", "Diploma"])
            department     = st.selectbox("Department",
                ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "Support"])

        with col2:
            job_level            = st.selectbox("Job Level", ["Junior", "Mid", "Senior", "Lead", "Manager"])
            experience_years     = st.slider("Experience (years)", 0, 35, 5)
            years_at_company     = st.slider("Years at Company", 0, 30, 3)
            salary               = st.number_input("Salary (₹/year)", 25000, 200000, 60000, step=5000)

        with col3:
            training_hours            = st.slider("Training Hours", 10, 100, 50)
            projects_completed        = st.slider("Projects Completed", 1, 20, 8)
            monthly_working_hours     = st.slider("Monthly Working Hours", 140, 220, 180)
            overtime_hours            = st.slider("Overtime Hours/month", 0, 40, 10)

        st.markdown("### 📋 Performance Indicators")
        col4, col5, col6 = st.columns(3)

        with col4:
            manager_rating         = st.slider("Manager Rating (1–5)", 1, 5, 3)
            leaves_taken           = st.slider("Leaves Taken (year)", 0, 20, 7)

        with col5:
            peer_review_score      = st.slider("Peer Review Score (0–10)", 0.0, 10.0, 6.5, step=0.1)
            attendance_rate        = st.slider("Attendance Rate (%)", 60.0, 100.0, 90.0, step=0.5)

        with col6:
            promotion_last_5_years = st.selectbox("Promoted in Last 5 Years?", [0, 1],
                                                   format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("🚀 Predict Performance")

    if submitted:
        try:
            from src.predictor import predict_single_employee
            emp_data = {
                "age": age, "gender": gender, "education": education,
                "department": department, "job_level": job_level,
                "experience_years": experience_years, "salary": salary,
                "training_hours": training_hours,
                "projects_completed": projects_completed,
                "monthly_working_hours": monthly_working_hours,
                "overtime_hours": overtime_hours,
                "leaves_taken": leaves_taken,
                "manager_rating": manager_rating,
                "peer_review_score": peer_review_score,
                "attendance_rate": attendance_rate,
                "years_at_company": years_at_company,
                "promotion_last_5_years": promotion_last_5_years,
            }
            result = predict_single_employee(emp_data, model=model)
            cat    = result["predicted_category"]

            st.markdown("---")
            st.markdown("## 📊 Prediction Results")

            r1, r2 = st.columns([1, 2])
            with r1:
                color_cls = {"High": "pred-high", "Medium": "pred-medium", "Low": "pred-low"}
                css_class = color_cls.get(cat, "")
                attrition_text = result["attrition_risk"]
                st.markdown("**Predicted Performance**")
                st.markdown(f"<div class='{css_class}'>&#9658; {cat}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"**Attrition Risk:** {attrition_text}")

            with r2:
                conf = result["confidence_%"]
                if conf:
                    fig_conf = go.Figure(go.Bar(
                        x=list(conf.values()), y=list(conf.keys()),
                        orientation="h",
                        marker=dict(color=["#A6E3A1","#FAB387","#F38BA8"],
                                    line=dict(width=0)),
                        text=[f"{v:.1f}%" for v in conf.values()],
                        textposition="outside"
                    ))
                    fig_conf.update_layout(
                        title="Confidence by Category (%)",
                        plot_bgcolor="#1E1E2E", paper_bgcolor="#1E1E2E",
                        template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0),
                        font=dict(color="#CDD6F4"), height=220
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)

            st.markdown(f"<div class='rec-box'>💡 <strong>HR Recommendation:</strong> {result['recommendation']}</div>",
                        unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}\nMake sure you have run `python main.py` first.")


# =============================================================
# PAGE 3: ANALYTICS DEEP DIVE
# =============================================================
elif page == "📈 Analytics Deep Dive":
    st.markdown('<div class="main-title">📈 Analytics Deep Dive</div>', unsafe_allow_html=True)
    df = load_data()

    st.subheader("Salary Distribution by Job Level")
    fig = px.box(df, x="job_level", y="salary",
                 color="job_level", template="plotly_dark",
                 color_discrete_sequence=px.colors.qualitative.Pastel,
                 category_orders={"job_level": ["Junior","Mid","Senior","Lead","Manager"]})
    fig.update_layout(plot_bgcolor="#1E1E2E", paper_bgcolor="#1E1E2E", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition Risk by Department")
        ar = df.groupby(["department","attrition_risk"]).size().reset_index(name="Count")
        fig2 = px.bar(ar, x="department", y="Count", color="attrition_risk",
                      color_discrete_map={"High":"#F38BA8","Medium":"#FAB387","Low":"#A6E3A1"},
                      template="plotly_dark", barmode="group")
        fig2.update_layout(plot_bgcolor="#1E1E2E", paper_bgcolor="#1E1E2E",
                           xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Performance by Education Level")
        edu = df.groupby("education")["performance_score"].mean().reset_index()
        edu.columns = ["Education","Avg Score"]
        fig3 = px.bar(edu, x="Education", y="Avg Score",
                      color="Avg Score", color_continuous_scale="Blues",
                      template="plotly_dark")
        fig3.update_layout(plot_bgcolor="#1E1E2E", paper_bgcolor="#1E1E2E", coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Correlation Heatmap (Numeric Features)")
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr().round(2)
    fig4 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     template="plotly_dark", aspect="auto", zmin=-1, zmax=1)
    fig4.update_layout(plot_bgcolor="#1E1E2E", paper_bgcolor="#1E1E2E")
    st.plotly_chart(fig4, use_container_width=True)

    # Raw data explorer
    st.subheader("📄 Raw Dataset Explorer")
    st.dataframe(df, use_container_width=True)


# =============================================================
# PAGE 4: ABOUT
# =============================================================
elif page == "ℹ️ About Project":
    st.markdown('<div class="main-title">ℹ️ About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    ## 🏢 Employee Performance Predictor using Data Analytics

    **Built by:** [Sarthak Dhumal]  
    **Purpose:** Student placement & internship portfolio project  
    **Domain:** HR Analytics | Machine Learning | Data Science

    ---

    ### 🎯 Problem Statement
    Companies struggle to:
    - Identify high-performing employees early
    - Detect flight risks before attrition occurs
    - Make data-driven promotion/training decisions

    ### 💡 Solution
    An AI-powered system that:
    - Analyzes 17+ employee attributes
    - Trains 5 ML models and picks the best
    - Predicts performance as **High / Medium / Low**
    - Provides actionable HR recommendations

    ---

    ### 🛠️ Tech Stack
    | Layer | Technology |
    |-------|-----------|
    | Language | Python 3.9+ |
    | ML Models | Random Forest, XGBoost, SVM, LR, GBM |
    | Data | Synthetic HR Dataset (1000 employees) |
    | Visualization | Plotly, Seaborn, Matplotlib |
    | Dashboard | Streamlit |
    | Storage | CSV + Joblib |

    ---

    ### 📁 Folder Structure
    ```
    Employee-Performance-Predictor/
    ├── data/          ← raw & processed datasets
    ├── src/           ← all Python modules
    ├── models/        ← saved ML models & encoders
    ├── images/        ← all EDA & model plots
    ├── outputs/       ← reports & predictions
    ├── main.py        ← pipeline entry point
    └── dashboard.py   ← this Streamlit dashboard
    ```

    ### 🚀 How to Run
    ```bash
    pip install -r requirements.txt
    python main.py
    streamlit run dashboard.py
    ```

    ### 📊 Model Performance
    > Run `python main.py` to see live results
    """)
