import joblib
import pandas as pd


MODEL_PATH = "models/logistic_regression_pipeline.joblib"


def main():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "checking_account_status": "A11",
        "duration_months": 24,
        "credit_history": "A32",
        "purpose": "A40",
        "credit_amount": 5000,
        "savings_account": "A61",
        "employment_since": "A72",
        "installment_rate": 4,
        "personal_status_sex": "A93",
        "other_debtors": "A101",
        "residence_since": 2,
        "property": "A123",
        "age": 30,
        "other_installment_plans": "A143",
        "housing": "A152",
        "existing_credits": 1,
        "job": "A173",
        "dependents": 1,
        "telephone": "A191",
        "foreign_worker": "A201",
    }])

    bad_risk_proba = model.predict_proba(sample)[:, 1][0]

    threshold = 0.45
    prediction = int(bad_risk_proba >= threshold)

    print("Bad risk probability:", round(bad_risk_proba, 4))
    print("Prediction:", prediction)
    print("Label:", "bad_risk" if prediction == 1 else "good_risk")


if __name__ == "__main__":
    main()