# Customer Revenue Risk & Churn Classification Model

## 📌 Business Problem
Quantifying Monthly Recurrence Revenue (MRR) loss by detecting customer churn drivers early using behavioral and support tracking data.

## 🛠️ Tech Stack & Methods
* **Language:** Python
* **Libraries:** `pandas`, `scikit-learn`, `statsmodels`, `matplotlib`
* **Statistical Analysis:** T-Test (continuous metrics) & Chi-Square Test (categorical flags)
* **Model:** Random Forest Classifier (`class_weight='balanced'`)

## 📊 Key Implementation Details & Results
* Performed hypothesis testing ($p < 0.05$) to prove churned users raising support tickets were statistically significant patterns rather than random noise.
* Handled severe 95/5 class imbalance by tuning model weights to prioritize minority churn risk.
* Generated dynamic churn probability scores to allow proactive customer success retention outreach.
