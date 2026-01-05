# =========================
# LIVE SINGLE PATIENT INPUT
# =========================

import numpy as np
import pandas as pd
import joblib

# -------------------------
# Load trained model & scaler
# -------------------------

model = joblib.load("xgboost.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------
# Feature list (must match training)
# -------------------------

features = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

# -------------------------
# Take input from user
# -------------------------

print("\nEnter patient medical values:\n")

patient_values = []

for feature in features:
    value = float(input(f"{feature}: "))
    patient_values.append(value)

# -------------------------
# Convert to DataFrame
# -------------------------

X_patient = pd.DataFrame([patient_values], columns=features)

# -------------------------
# Scale input
# -------------------------

X_patient_scaled = scaler.transform(X_patient)

# -------------------------
# Predict
# -------------------------

prediction = model.predict(X_patient)[0]
probability = model.predict_proba(X_patient)[0][1]

# -------------------------
# Output result
# -------------------------

print("\n--- Diagnosis Result ---")

if prediction == 1:
    print("Diagnosis: Benign (No Cancer)")
else:
    print("Diagnosis: Malignant (Cancer Detected)")

print(f"Risk Probability: {probability:.2f}")
