import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Generate Synthetic Customer Dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame(
    {
        "customer_id": range(1000, 1000 + n_samples),
        "monthly_charges": np.random.uniform(20, 120, n_samples),
        "support_tickets": np.random.poisson(lam=2, size=n_samples),
        "tenure_months": np.random.randint(1, 60, n_samples),
        "churn": np.random.choice([0, 1], size=n_samples, p=[0.90, 0.10]),
    }
)

# 2. Hypothesis Testing (T-Test on Support Tickets vs Churn)
churned = data[data["churn"] == 1]["support_tickets"]
retained = data[data["churn"] == 0]["support_tickets"]
t_stat, p_val = stats.ttest_ind(churned, retained)
print(f"T-Statistic: {t_stat:.4f}, p-value: {p_val:.4f}")

# 3. Model Training (Random Forest with Imbalance Handling)
X = data[["monthly_charges", "support_tickets", "tenure_months"]]
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

print(
    "Model Training Complete. Accuracy Score:", clf.score(X_test, y_test)
)
