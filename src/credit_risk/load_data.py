import pandas as pd
from ucimlrepo import fetch_ucirepo


COLUMN_NAMES = {
    "Attribute1": "checking_account_status",
    "Attribute2": "duration_months",
    "Attribute3": "credit_history",
    "Attribute4": "purpose",
    "Attribute5": "credit_amount",
    "Attribute6": "savings_account",
    "Attribute7": "employment_since",
    "Attribute8": "installment_rate",
    "Attribute9": "personal_status_sex",
    "Attribute10": "other_debtors",
    "Attribute11": "residence_since",
    "Attribute12": "property",
    "Attribute13": "age",
    "Attribute14": "other_installment_plans",
    "Attribute15": "housing",
    "Attribute16": "existing_credits",
    "Attribute17": "job",
    "Attribute18": "dependents",
    "Attribute19": "telephone",
    "Attribute20": "foreign_worker",
}


def load_german_credit():
    german_credit = fetch_ucirepo(id=144)

    X = german_credit.data.features
    y = german_credit.data.targets

    target_col = y.columns[0]

    df = X.copy()
    df = df.rename(columns=COLUMN_NAMES)

    df["target_original"] = y[target_col]
    df["bad_risk"] = df["target_original"].map({
        1: 0,
        2: 1
    })

    return df


if __name__ == "__main__":
    df = load_german_credit()

    print("Shape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nBad risk distribution:")
    print(df["bad_risk"].value_counts())

    print("\nBad risk distribution (%):")
    print(df["bad_risk"].value_counts(normalize=True))