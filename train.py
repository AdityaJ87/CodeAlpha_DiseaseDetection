# =========================
# TRAINING SCRIPT
# =========================

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------
# Load Dataset
# -------------------------

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# -------------------------
# Train-Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# Scaling (for LR & SVM)
# -------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# -------------------------
# Models
# -------------------------

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svm": SVC(kernel="rbf", probability=True),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgboost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

# -------------------------
# Train & Save Models
# -------------------------

for name, model in models.items():
    
    if name in ["logistic_regression", "svm"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    
    joblib.dump(model, f"{name}.pkl")
    print(f"Model saved: {name}.pkl")

print("\nTraining completed successfully.")
