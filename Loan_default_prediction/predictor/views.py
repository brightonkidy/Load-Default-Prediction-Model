from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
from pathlib import Path
import joblib
import numpy as np


MODEL_DIR = Path(__file__).resolve().parent.parent / 'Loan_default_prediction' / 'model'
MODEL_CANDIDATES = [
    'random_forest_model.pkl',
    'model.pkl',
]

# Feature names in the exact order expected by the model
FEATURE_NAMES = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
    'Property_Area',
]


def predict_view(request: HttpRequest) -> HttpResponse:
    """
    One-page form with labeled fields; encodes inputs to model-ready vector.
    Feature order:
    [Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
     CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]
    """
    context = {"prediction": None, "error": None}

    if request.method == 'POST':
        try:
            # Read fields as strings, coerce to floats/ints (simple encoding matches guidance)
            gender = int(request.POST.get('gender'))
            married = int(request.POST.get('married'))
            dependents_raw = request.POST.get('dependents')
            dependents = 3 if dependents_raw == '3' else int(dependents_raw)
            education = int(request.POST.get('education'))
            self_employed = int(request.POST.get('self_employed'))
            applicant_income = float(request.POST.get('applicant_income'))
            coapplicant_income = float(request.POST.get('coapplicant_income'))
            loan_amount = float(request.POST.get('loan_amount'))
            loan_amount_term = float(request.POST.get('loan_amount_term'))
            credit_history = float(request.POST.get('credit_history'))
            property_area = int(request.POST.get('property_area'))

            feature_values = [
                gender,
                married,
                dependents,
                education,
                self_employed,
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_amount_term,
                credit_history,
                property_area,
            ]
            X = np.array([feature_values], dtype=float)

            existing_path = None
            for name in MODEL_CANDIDATES:
                candidate = MODEL_DIR / name
                if candidate.exists():
                    existing_path = candidate
                    break

            if existing_path is None:
                raise FileNotFoundError(
                    f"Model not found. Checked: " + ", ".join(str(MODEL_DIR / n) for n in MODEL_CANDIDATES)
                )

            loaded_obj = joblib.load(existing_path)

            # If a dict was saved, try to extract the estimator
            if isinstance(loaded_obj, dict):
                candidate = None
                # Common keys first
                for key in ['model', 'estimator', 'rf', 'pipeline', 'clf']:
                    if key in loaded_obj and hasattr(loaded_obj[key], 'predict'):
                        candidate = loaded_obj[key]
                        break
                # Otherwise scan values
                if candidate is None:
                    for value in loaded_obj.values():
                        if hasattr(value, 'predict'):
                            candidate = value
                            break
                if candidate is None:
                    raise TypeError('Loaded object is a dict and no estimator with predict() was found.')
                model = candidate
            else:
                model = loaded_obj
            y_pred = model.predict(X)[0]
            try:
                y_proba = getattr(model, 'predict_proba')(X)[0, int(y_pred)] if hasattr(model, 'predict_proba') else None
            except Exception:
                y_proba = None

            # Map raw label to human-friendly text
            raw_label = int(y_pred) if isinstance(y_pred, (np.integer, int)) else y_pred
            label_text = None
            try:
                classes = getattr(model, 'classes_', None)
                if classes is not None and len(classes) == 2:
                    # Handle common encodings
                    if all(isinstance(c, str) for c in classes):
                        # Typical ['N','Y'] or ['Y','N']
                        label_text = 'Approved' if str(y_pred).upper().startswith('Y') else 'Not Approved'
                    else:
                        # Numeric classes, assume 1=Approved, 0=Not Approved
                        label_text = 'Approved' if int(y_pred) == 1 else 'Not Approved'
            except Exception:
                label_text = None
            if label_text is None:
                # Fallback
                label_text = 'Approved' if raw_label in (1, 'Y', 'y', 'Yes', 'YES') else 'Not Approved'

            # Attempt to compute reasons for the decision
            reasons = []
            try:
                # Prefer SHAP if available (tree models)
                import shap  # type: ignore
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(X)
                    # shap_values may be list for classifiers: pick predicted class
                    if isinstance(shap_vals, list):
                        shap_for_pred = shap_vals[int(raw_label)]
                    else:
                        shap_for_pred = shap_vals
                    contribs = shap_for_pred[0]
                    pairs = list(zip(FEATURE_NAMES, contribs, X[0]))
                    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
                    for name, contrib, val in pairs[:3]:
                        direction = 'increased' if contrib > 0 else 'decreased'
                        reasons.append(f"{name} {direction} the likelihood (value: {val})")
                except Exception:
                    # Fallback to feature_importances_
                    importances = getattr(model, 'feature_importances_', None)
                    if importances is not None and len(importances) == len(FEATURE_NAMES):
                        pairs = list(zip(FEATURE_NAMES, importances, X[0]))
                        pairs.sort(key=lambda t: t[1], reverse=True)
                        for name, imp, val in pairs[:3]:
                            reasons.append(f"{name} had strong influence (value: {val})")
            except Exception:
                # SHAP not installed or error; ignore reasons
                pass

            # If still no reasons (e.g., features mismatch), use local sensitivity analysis
            if (not reasons) and hasattr(model, 'predict_proba'):
                try:
                    base_proba = float(model.predict_proba(X)[0, 1]) if hasattr(model, 'classes_') else float(model.predict_proba(X)[0, -1])
                    deltas = []
                    x0 = X[0].astype(float)
                    for idx, (name, val) in enumerate(zip(FEATURE_NAMES, x0)):
                        x_variant = x0.copy()
                        # Try multiple candidate changes and pick the one that most increases approval probability
                        candidates = []
                        if name in ('Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History'):
                            candidates = [1.0 - float(val)]
                        elif name == 'Dependents':
                            candidates = [max(0.0, val - 1.0), min(3.0, val + 1.0)]
                        elif name in ('ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'):
                            candidates = [max(0.0, val * 0.8), max(0.0, val * 1.2)]
                        elif name == 'Loan_Amount_Term':
                            candidates = [max(0.0, val - 12.0), val + 12.0]
                        elif name == 'Property_Area':
                            candidates = [c for c in (0.0, 1.0, 2.0) if c != val]

                        best_change = None
                        best_new = val
                        for cand in candidates:
                            x_tmp = x0.copy(); x_tmp[idx] = cand
                            proba_variant = float(model.predict_proba([x_tmp])[0, 1])
                            change = proba_variant - base_proba
                            if best_change is None or change > best_change:
                                best_change = change; best_new = cand

                        if best_change is None:
                            continue
                        deltas.append((name, float(best_change), float(val), float(best_new)))

                    deltas.sort(key=lambda t: abs(t[1]), reverse=True)

                    def pretty_area(v: float) -> str:
                        return {0.0: 'Rural', 1.0: 'Semiurban', 2.0: 'Urban'}.get(round(v), str(v))

                    for name, change, old_val, new_val in deltas[:3]:
                        # Craft applicant-centric messages
                        if name == 'ApplicantIncome':
                            if new_val > old_val and change > 0:
                                reasons.append(f"Not approved because applicant income is relatively low (entered {old_val}). Higher income improves approval.")
                            elif new_val < old_val and change > 0:
                                reasons.append(f"Not approved; a lower income improves approval pattern, indicating income-profile mismatch (entered {old_val}).")
                            else:
                                reasons.append(f"Applicant income (entered {old_val}) influenced the decision.")
                        elif name == 'CoapplicantIncome':
                            if new_val > old_val and change > 0:
                                reasons.append(f"Co-applicant income appears low (entered {old_val}). Higher co-applicant income would help.")
                            else:
                                reasons.append(f"Co-applicant income (entered {old_val}) influenced the decision.")
                        elif name == 'LoanAmount':
                            if new_val < old_val and change > 0:
                                reasons.append(f"Requested loan amount is relatively high (entered {old_val}). A lower amount increases approval likelihood.")
                            elif new_val > old_val and change > 0:
                                reasons.append(f"A higher loan amount improves approval for this profile, but current amount (entered {old_val}) affected the decision.")
                            else:
                                reasons.append(f"Loan amount (entered {old_val}) influenced the decision.")
                        elif name == 'Loan_Amount_Term':
                            if new_val > old_val and change > 0:
                                reasons.append(f"Longer repayment term could improve affordability (entered {old_val} months).")
                            elif new_val < old_val and change > 0:
                                reasons.append(f"Shorter repayment term could improve approval (entered {old_val} months).")
                            else:
                                reasons.append(f"Loan term (entered {old_val} months) influenced the decision.")
                        elif name == 'Credit_History':
                            if new_val > old_val and change > 0:
                                reasons.append("No or limited credit history reduced approval chances. A positive credit history would help.")
                            elif new_val < old_val and change > 0:
                                reasons.append("Credit history pattern reduced approval chances.")
                            else:
                                reasons.append("Credit history influenced the decision.")
                        elif name == 'Dependents':
                            if new_val < old_val and change > 0:
                                reasons.append(f"Many dependents (entered {int(old_val)}) may reduce affordability. Fewer dependents would help.")
                            else:
                                reasons.append(f"Number of dependents (entered {int(old_val)}) influenced the decision.")
                        elif name == 'Property_Area':
                            if change > 0 and new_val != old_val:
                                reasons.append(f"Applicants from {pretty_area(new_val)} areas tend to be approved more than {pretty_area(old_val)} for similar profiles.")
                            else:
                                reasons.append(f"Property area ({pretty_area(old_val)}) influenced the decision.")
                        elif name in ('Gender','Married','Education','Self_Employed'):
                            reasons.append(f"Applicant profile factor '{name}' influenced the decision.")
                except Exception:
                    pass

            context["prediction"] = {
                "label": raw_label,
                "label_text": label_text,
                "probability": float(y_proba) if y_proba is not None else None,
                "reasons": reasons,
            }
        except Exception as exc:
            context["error"] = str(exc)

    return render(request, 'predictor/predict.html', context)


def docs_view(request: HttpRequest) -> HttpResponse:
    """Render project documentation with a print-to-PDF friendly layout."""
    try:
        readme_path = Path(__file__).resolve().parent.parent / 'README.md'
        readme_text = readme_path.read_text(encoding='utf-8') if readme_path.exists() else 'README.md not found.'
    except Exception as exc:
        readme_text = f'Error reading README.md: {exc}'
    return render(request, 'predictor/docs.html', {"readme": readme_text})
