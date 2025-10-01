# Load-Default-Prediction-Model
This project creates a loan default prediction model for ZWMB, using applicant data like income, employment, and credit history to identify high-risk borrowers. It offers a data-driven, objective way to assess credit risk, helping reduce financial losses and improve lending decisions.

### 1) Project Structure
```text
Loan_default_prediction/
├─ manage.py
├─ Loan_default_prediction/                # Django project settings
│  ├─ __init__.py
│  ├─ asgi.py
│  ├─ settings.py
│  ├─ urls.py
│  ├─ wsgi.py
│  ├─ data/
│  │  └─ Dataset.csv
│  └─ model/
│     ├─ Loan_Approval_Prediction (1).ipynb
│     └─ model.pkl (or random_forest_model.pkl)
└─ predictor/                              # App: UI + prediction logic
   ├─ __init__.py
   ├─ admin.py
   ├─ apps.py
   ├─ migrations/
   ├─ models.py
   ├─ views.py                             # Loads model and predicts
   ├─ urls.py
   └─ templates/
      └─ predictor/
         └─ predict.html                   # One-page form + results
```

### 2) User Manual (How to Run and Use)
- Prerequisites:
  - Python 3.10+
  - pip and virtualenv (recommended)
- Setup:
  ```bash
  cd Loan_default_prediction
  python -m venv .venv
  .venv/Scripts/activate  # Windows PowerShell
  pip install -r requirements.txt
  python manage.py migrate
  python manage.py runserver
  ```
- Train and save the model from the notebook:
  ```python
  # After training RandomForest as model3
  import joblib
  joblib.dump(model3, 'model.pkl')  # saved into Loan_default_prediction/Loan_default_prediction/model/
  ```
- Open the app: visit http://127.0.0.1:8000/
- Fill the labeled form fields and press Predict.
- The result card shows Decision, Raw label, Confidence, and (if available) Reasons.

Notes:
- Place your logo at `predictor/static/predictor/logo.png` if you want the header logo to render.
- If static files don’t appear, hard-refresh the page. In production set DEBUG=False and run `python manage.py collectstatic`.

### 3) Tools Used to Develop the Model and App
- Model and data science:
  - scikit-learn (RandomForestClassifier)
  - numpy, pandas
  - joblib (model persistence)
  - Optional: shap (per-prediction explanations)
  - Optional during training: imbalanced-learn (resampling) if used
- Web application:
  - Django 4.x (views, templates, static files)

### 4) Requirements Engineering
- Functional requirements:
  - Load a persisted ML model and serve predictions via a web page.
  - Provide a user-friendly form with labeled fields (dropdowns + numeric inputs).
  - Display decision, confidence score, and short reasons for the decision.
  - Allow easy theming (ZWMB colors, logo) to match branding.
- Non-functional requirements:
  - Usability: understandable labels, guidance and one-page interaction.
  - Maintainability: model file paths and loaders are robust to common filenames.
  - Reliability: clear error messages when model file is missing or incompatible.
  - Performance: single-request inference with lightweight model loading.
  - Security: CSRF protection on the form; no PII stored server-side.

### 5) Objectives of the Model – and What Was Met
- Predict Loan Default Risk
- Improve Lending Decisions
- Reduce Non-Performing Loans
- Enhance Customer Profiling
- Generate Prediction decision reasons.

Status: All above objectives were implemented in the app; explanations are available (best with SHAP installed) and a heuristic fallback is provided.

### 6) Brief Description of the Model Built
- Algorithm: RandomForestClassifier (ensemble of decision trees) trained on features such as Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area.
- Persistence: saved with joblib to `model.pkl` (or `random_forest_model.pkl`).
- Inference: numeric vector in the fixed feature order is passed to the model; the backend maps form fields to that order.
- Explanations: if `shap` is available, SHAP values provide top per-feature contributions; otherwise a feature importance or local sensitivity fallback summarizes the strongest drivers.

### 7) Problems Identified and Considerations
- Model persistence pitfalls: saving a Python dict instead of the estimator caused `dict has no attribute predict` – save the estimator directly.
- Notebook `__file__` is undefined: when saving from notebooks use `Path.cwd()` or a relative filename.
- Feature encoding consistency: ensure the form encodes categories the same way as in training; mismatches reduce accuracy.
- Static file path: logos and static assets must be placed under `predictor/static/predictor/`.
- Data quality: dataset contains missing values and potential outliers; preprocessing is important.
- Class imbalance: approvals vs. non-approvals may be imbalanced; consider resampling or class weights.
- Fairness and explainability: predictions might embed historical bias; monitor feature impacts and consider adding policy constraints.

### 8) Requirements.txt (sample)
If you don’t already have one, create `requirements.txt` similar to:
```text
Django==4.2.7
scikit-learn==1.3.2
numpy==1.26.4
pandas==2.1.4
joblib==1.3.2
shap==0.45.0  # optional, for explanations
```
