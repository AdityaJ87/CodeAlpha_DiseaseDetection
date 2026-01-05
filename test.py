# =========================
# TESTING SCRIPT
# =========================

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# -------------------------
# Load Dataset
# -------------------------

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# -------------------------
# Load Scaler
# -------------------------

scaler = joblib.load("scaler.pkl")

# -------------------------
# Load Models
# -------------------------

models = {
    "logistic_regression": joblib.load("logistic_regression.pkl"),
    "svm": joblib.load("svm.pkl"),
    "random_forest": joblib.load("random_forest.pkl"),
    "xgboost": joblib.load("xgboost.pkl")
}

# -------------------------
# Prepare Data
# -------------------------

X_scaled = scaler.transform(X)

# -------------------------
# Evaluate Models
# -------------------------

for name, model in models.items():
    
    if name in ["logistic_regression", "svm"]:
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
    
    print(f"\n===== {name.upper()} =====")
    print("Accuracy :", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall   :", recall_score(y, y_pred))
    print("F1 Score :", f1_score(y, y_pred))
    print("ROC-AUC  :", roc_auc_score(y, y_prob))
    print(classification_report(y, y_pred))

# -------------------------
# Single Patient Prediction
# -------------------------

sample = X.iloc[0].values.reshape(1, -1)
sample_scaled = scaler.transform(sample)

model = models["xgboost"]

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("\nSingle Patient Prediction:")
print("Result:", "Benign" if prediction == 1 else "Malignant")
print("Probability:", probability)
