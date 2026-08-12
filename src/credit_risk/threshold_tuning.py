import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.load_data import load_german_credit


def main():
    df = load_german_credit()

    X = df.drop(columns=["target_original", "bad_risk"])
    y = df["bad_risk"]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)

    bad_risk_proba = model.predict_proba(X_test)[:, 1]

    thresholds = [i / 100 for i in range(10, 91, 5)]

    rows = []

    for threshold in thresholds:
        y_pred = (bad_risk_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        business_cost = fp * 1 + fn * 5

        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "business_cost": business_cost,
        })

    results = pd.DataFrame(rows)

    print("\nThreshold tuning results:")
    print(results)

    print("\nBest threshold by business cost:")
    print(results.sort_values("business_cost").head(5))

    results.to_csv("reports/threshold_tuning_results.csv", index=False)


if __name__ == "__main__":
    main()