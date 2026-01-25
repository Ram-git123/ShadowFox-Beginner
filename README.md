# ShadowFox-Beginner
"An end-to-end Machine Learning masterpiece using Gradient Boosting to predict Boston housing prices. Features include leak-proof Scikit-Learn pipelines, Bayesian optimization (Optuna), and Model Interpretability (SHAP)."

# 🏠 Boston Housing: End-to-End Production ML Pipeline

This project transforms the classic Boston Housing task into a **Production-Ready Machine Learning System**. Beyond simple prediction, it implements a robust architecture focusing on **Data Integrity**, **Automated Optimization**, and **Model Interpretability**.

## 🎯 Key Project Highlights

* **🏆 Professional Pipeline:** 100% encapsulation of preprocessing, feature engineering, and modeling using `Scikit-Learn Pipelines` to prevent **Data Leakage**.
* **🛠️ Advanced Feature Engineering:** Developed domain-specific features like `RM_LSTAT_ratio` (spatial-socio interaction) and `Tax_per_Room` (ownership efficiency).
* **🧪 Bayesian Optimization:** Utilized **Optuna** for hyperparameter tuning, moving beyond brute-force searches to find the optimal "Goldilocks" zone for model performance.
* **🧠 Explainable AI (XAI):** Integrated **SHAP (Shapley Additive Explanations)** to deconstruct "Black Box" decisions into transparent, feature-level insights.

## 🏗️ Architecture & Workflow

### 1. Data Audit & EDA

The initial audit identified **20 missing values** per column and a **target skewness of ~1.11**. We addressed this through **Median Imputation** and statistical scaling.

### 2. Feature Engineering Logic

We engineered features to capture non-linear relationships:

* **`RM_LSTAT_ratio`**: Captures the quality of living space relative to the neighborhood's socio-economic status.
* **`Tax_per_Room`**: Normalizes the property tax burden against house size.

### 3. Model Selection (The Shootout)

We compared four mathematical architectures using 5-fold Cross-Validation:

1. **Ridge Regression** (Linear Baseline)
2. **Random Forest** (Bagging)
3. **Gradient Boosting** (Boosting - **WINNER**)

### 4. Hyperparameter Tuning (Optuna)

Instead of a Grid Search, we used Bayesian Search to optimize:

* `learning_rate`: Log-scaled search for fine-tuned convergence.
* `n_estimators` & `max_depth`: Optimized to balance bias and variance.

## 🧠 Model Interpretability (SHAP)

We used SHAP values to ensure the model remains economically sound.

* **Global Importance:** Confirmed **Average Rooms (RM)** and **Lower Status Population (LSTAT)** as the primary drivers.
* **Dependence Analysis:** Identified specific inflection points where neighborhood status begins to exponentially impact price.

## 📦 Deployment & Usage

The entire "brain" of the project is serialized in `masterpiece_boston_model.pkl`. This allows for **one-click inference** on raw data.

### How to use the Model:

``python
import joblib

# 1. Load the production-ready pipeline
model = joblib.load('models/masterpiece_boston_model.pkl')

# 2. Predict instantly (preprocessing is handled automatically!)
predictions = model.predict(new_data)'''

