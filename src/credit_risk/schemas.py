from pydantic import BaseModel, Field


class CreditRiskInput(BaseModel):
    checking_account_status: str
    duration_months: int = Field(gt=0)
    credit_history: str
    purpose: str
    credit_amount: int = Field(gt=0)
    savings_account: str
    employment_since: str
    installment_rate: int = Field(ge=1, le=4)
    personal_status_sex: str
    other_debtors: str
    residence_since: int = Field(ge=1, le=4)
    property: str
    age: int = Field(gt=0)
    other_installment_plans: str
    housing: str
    existing_credits: int = Field(ge=1)
    job: str
    dependents: int = Field(ge=1)
    telephone: str
    foreign_worker: str


class CreditRiskPrediction(BaseModel):
    bad_risk_probability: float
    threshold: float
    prediction: int
    risk_label: str