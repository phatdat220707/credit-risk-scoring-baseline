import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = "models/logistic_regression_pipeline.joblib"
THRESHOLD = 0.45

app = FastAPI(title="Credit Risk Scoring API")
model = joblib.load(MODEL_PATH)


class CreditRiskInput(BaseModel):
    checking_account_status: str
    duration_months: int = Field(gt=0)
    credit_history: str
    purpose: str
    credit_amount: int = Field(gt=0)
    savings_account: str
    employment_since: str
    installment_rate: int = Field(ge=1,le=4)
    personal_status_sex: str
    other_debtors: str
    residence_since: int = Field(ge=1,le=4)
    property: str
    age: int = Field(gt=0)
    other_installment_plans: str
    housing: str
    existing_credits: int = Field(ge=1)
    job: str
    dependents: int = Field(ge=1)
    telephone: str
    foreign_worker: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "Credit Risk Scoring API",
        "endpoints": ["/health", "/predict"],
    }


@app.post("/predict")
def predict_credit_risk(payload: CreditRiskInput):
    input_df = pd.DataFrame([payload.model_dump()])

    bad_risk_probability = model.predict_proba(input_df)[:, 1][0]
    prediction = int(bad_risk_probability >= THRESHOLD)
    risk_label = "bad_risk" if prediction == 1 else "good_risk"

    return {
        "bad_risk_probability": round(float(bad_risk_probability), 4),
        "threshold": THRESHOLD,
        "prediction": prediction,
        "risk_label": risk_label,
    }