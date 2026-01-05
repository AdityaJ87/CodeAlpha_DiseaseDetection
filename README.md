Disease Prediction from Medical Data

Overview

This project implements a machine learning–based disease prediction system that predicts the likelihood of disease based on structured medical data. The system is designed to support early diagnosis using supervised classification algorithms.

Dataset Used

Breast Cancer Wisconsin (Diagnostic) Dataset

Source: UCI Machine Learning Repository

Accessed via: sklearn.datasets.load_breast_cancer()

Dataset Details:

Total samples: 569 patients

Features: 30 medical features

Target classes:

0 → Malignant

1 → Benign

Each row represents a single patient’s clinical measurements.

Algorithms Implemented

Logistic Regression

Support Vector Machine (SVM)

Random Forest

XGBoost

Project Structure
CodeAlpha_DiseaseDetection/
│
├── train_model.py          # Model training & saving
├── test_model.py           # Model evaluation
├── single_patient_input.py # Live patient input prediction
├── scaler.pkl              # Saved scaler
├── *.pkl                   # Trained ML models
└── README.md

Features

End-to-end ML pipeline

Separate training and testing scripts

Live single-patient prediction

Probability-based medical risk output

Production-ready structure

Disclaimer

This system is intended for educational and research purposes only.
It does not replace professional medical diagnosis.
