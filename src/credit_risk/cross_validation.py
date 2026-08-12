import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.load_data import load_german_credit


def main():
    df = load_german_credit()

    X = df.drop(columns=["target_original", "bad_risk"])
    y = df["bad_risk"]

    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"],
    )

    results = pd.DataFrame(scores)
    summary = results.agg(["mean", "std"]).T

    print("\nCross-validation results:")
    print(summary)

    summary.to_csv("reports/cross_validation_results.csv")


if __name__ == "__main__":
    main()