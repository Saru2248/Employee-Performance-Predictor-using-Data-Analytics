# Employee Performance Predictor — Interview Preparation Guide
# =============================================================

## 🎯 How to Explain This Project

### To HR (Non-Technical Interviewer)
> "I built a system that uses historical employee data — like attendance,
> training hours, project completions, and manager ratings — to predict
> whether an employee will be a High, Medium, or Low performer.
> This helps HR teams make smarter decisions about promotions, training
> investments, and identifying employees who might be at risk of leaving.
> Think of it like a health check-up, but for employee performance."

### To Technical Interviewer
> "I built an end-to-end ML classification pipeline using Python and
> scikit-learn. I generated a synthetic HR dataset of 1,000 employees with
> 17 features. After EDA and feature engineering — including composite
> scores like engagement score and salary-per-experience — I trained 5
> models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost,
> and SVM. I used stratified 5-fold cross-validation for model selection,
> evaluated on precision, recall, F1 score, and ROC-AUC, and deployed
> the best model via a Streamlit dashboard with real-time predictions."

---

## ❓ Interview Questions & Strong Answers

### Q1: Why did you use synthetic data instead of real data?
**Answer:**
Real HR datasets contain sensitive PII (Personally Identifiable Information)
and are protected by privacy laws like GDPR and India's PDPB. As a student,
I don't have access to corporate HR databases. Instead, I designed synthetic
data using realistic statistical distributions (normal distributions for
scores, categorical probabilities for departments/levels) — ensuring the
data mimics real-world patterns while being completely ethical to use.

---

### Q2: How did you choose between Random Forest and XGBoost?
**Answer:**
I compared models using two criteria:
1. **5-fold stratified cross-validation accuracy** on the training set
2. **Test accuracy** and **weighted F1 score** on the held-out 20% test set

Random Forest tends to be more robust to outliers and noise. XGBoost is
generally more accurate on tabular data due to boosting. I let the
evaluation decide — whichever scored higher on both metrics was selected
as the "best model". This is the industry-standard model selection process.

---

### Q3: What is stratified K-Fold and why use it?
**Answer:**
Our target variable (High/Medium/Low) has imbalanced class distribution.
Regular K-Fold might put all "Low" performers in one fold, making training
unreliable. Stratified K-Fold ensures each fold has the same proportion of
each class as the full dataset — making evaluation more reliable.

---

### Q4: Explain feature engineering decisions.
**Answer:**
Raw features alone (like salary and experience) don't tell the whole story.
I created:
- **salary_per_exp**: Detects if someone is underpaid for their experience
- **engagement_score**: Composite of attendance, training, peer review
- **overtime_ratio**: High overtime = burnout risk = performance dip
- **projects_per_year**: Productivity metric
- **is_senior**: Binary flag for senior/lead/manager roles

These engineered features capture business logic that raw columns miss.

---

### Q5: How would you handle class imbalance?
**Answer:**
In this dataset, Medium performers are the majority. I used:
1. `class_weight='balanced'` in sklearn models (adjusts loss function)
2. **Stratified splitting** to preserve class ratios
3. **Weighted F1 score** as evaluation metric (penalizes majority-class bias)
If imbalance were severe, I'd also use SMOTE (Synthetic Minority Oversampling)
from the imbalanced-learn library.

---

### Q6: What is ROC-AUC and why is it useful here?
**Answer:**
ROC-AUC (Receiver Operating Characteristic – Area Under Curve) measures how
well the model distinguishes between classes regardless of the classification
threshold. A value of 1.0 = perfect, 0.5 = random guess. In a multi-class
setting, I use the "one-vs-rest" approach. For HR, this matters because the
cost of misclassifying a High performer as Low (false negative) is very
different from the reverse — ROC-AUC captures this nuance better than accuracy.

---

### Q7: How is this system useful to a real company?
**Answer:**
Companies using this system can:
- **Promote proactively**: Identify High performers before they resign
- **Reduce training waste**: Focus training budgets on Medium performers
  most likely to improve
- **Retention programs**: Flag High-risk attrition employees for retention
  bonuses or role changes
- **Performance Improvement Plans (PIP)**: Target Low performers early
- **Unbiased decisions**: Remove subjective bias from promotion decisions

---

### Q8: What would be your next steps to improve this project?
**Answer:**
1. **Real data integration**: Connect to HRMS APIs (SAP SuccessFactors, Darwinbox)
2. **Deep Learning**: Try a TabNet or Transformer-based model for tabular data
3. **Attrition prediction** as a second model (binary classification)
4. **SHAP explainability**: Show which features drove each prediction
5. **Time-series tracking**: Predict performance trajectory over time
6. **Real-time inference API**: FastAPI microservice with Docker deployment

---

### Q9: Explain the business logic behind your performance score formula.
**Answer:**
I weighted 6 key HR factors:
- Manager Rating (25%): Most direct measure of perceived performance
- Peer Review (20%): 360-degree feedback component
- Attendance (15%): Baseline reliability metric
- Training hours (15%): Investment in self-development
- Projects completed (15%): Quantifiable output
- Work-life balance (10%): Overtime inversely impacts long-term performance

These weights are based on real HR frameworks like OKRs and 360-degree
feedback systems used by companies like Google and Deloitte.

---

### Q10: What is the difference between precision and recall?
**Answer:**
- **Precision**: Of all employees predicted as "Low performer," how many
  actually are? (Avoid falsely labeling good employees as poor)
- **Recall**: Of all actual "Low performers," how many did we correctly catch?
  (Critical for retention — don't miss at-risk employees)
- **F1 Score**: Harmonic mean of both — balances the trade-off

In HR, recall for "Low" is often more critical (catching all struggling
employees) while precision matters for "High" (promoting the right people).

---

## 📸 Screenshots to Capture for Proof

1. **Dataset preview** — `df.head()` showing 21 columns
2. **EDA — Performance distribution bar chart**
3. **EDA — Correlation heatmap**
4. **EDA — Training hours vs performance scatter**
5. **Model comparison bar chart** (all 5 models)
6. **Confusion matrix** of best model
7. **Feature importance** top 15
8. **Terminal output** showing accuracy & F1 score
9. **Streamlit dashboard** — Overview page
10. **Streamlit dashboard** — Prediction result for sample employee

---

## 🔮 Future Improvements

| Upgrade | Description | Difficulty |
|---------|-------------|-----------|
| Real HR API | Connect to SAP/Darwinbox APIs | Advanced |
| SHAP Values | Explain individual predictions | Intermediate |
| Employee Attrition Model | Predict who will resign | Intermediate |
| Deep Learning | TabNet for tabular data | Advanced |
| Streamlit Cloud Deploy | Host dashboard publicly | Easy |
| CI/CD Pipeline | GitHub Actions automation | Intermediate |
| Real-time System | FastAPI + Docker microservice | Advanced |

---

## 🛠️ Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| ModuleNotFoundError: xgboost | Not installed | `pip install xgboost` |
| FileNotFoundError: employee_data.csv | main.py not run yet | `python main.py` first |
| No trained model found | models/ folder empty | `python main.py` first |
| Streamlit port in use | Another app running | `streamlit run dashboard.py --server.port 8502` |
| ValueError: unknown category | New category in test | Retrain or use handle_unknown='ignore' |
| ConvergenceWarning (LogReg) | Too few iterations | Already set max_iter=1000 |
| Memory error on large dataset | Too much RAM | Reduce N_EMPLOYEES or use chunked reading |
| UnicodeDecodeError | CSV encoding issue | Add `encoding='utf-8-sig'` to pd.read_csv |
