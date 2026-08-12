import joblib
import pandas as pd

from fastapi import FastAPI

from credit_risk.schemas import CreditRiskInput, CreditRiskPrediction

from pathlib import Path

from fastapi.responses import HTMLResponse

MODEL_PATH = Path("models/logistic_regression_pipeline.joblib")
THRESHOLD = 0.45

app = FastAPI(title="Credit Risk Scoring API")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Run save_model.py first."
    )

model = joblib.load(MODEL_PATH)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "Credit Risk Scoring API",
        "endpoints": ["/health", "/predict"],
    }

@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return """
    <html>
        <head><title>Credit Risk Demo</title></head>
        <body style="font-family: Arial; max-width: 800px; margin: 40px auto;">
            <h1>Credit Risk Scoring Demo</h1>
            <p>Open <a href="/docs">/docs</a> to test the prediction API.</p>
            <p>Use POST /predict with applicant data to get bad risk probability.</p>
        </body>
    </html>
    """

@app.post("/predict", response_model=CreditRiskPrediction)
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