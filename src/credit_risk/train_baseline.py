import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.load_data import load_german_credit


def evaluate_model(name, y_true, y_pred, y_proba):
    print(f"\n{name}")
    print("-" * len(name))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_true, y_proba))
    print("PR-AUC:", average_precision_score(y_true, y_proba))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def main():
    df = load_german_credit()

    X = df.drop(columns=["target_original", "bad_risk"])
    y = df["bad_risk"]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    print("Categorical features:", categorical_features)
    print("Numeric features:", numeric_features)

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

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    random_forest_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)
    rf_model = random_forest_model.fit(X_train, y_train)

    logistic_pred = logistic_model.predict(X_test)
    logistic_proba = logistic_model.predict_proba(X_test)[:, 1]

    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    evaluate_model("Logistic Regression", y_test, logistic_pred, logistic_proba)
    evaluate_model("Random Forest", y_test, rf_pred, rf_proba)


if __name__ == "__main__":
    main()