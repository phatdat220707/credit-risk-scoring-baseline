"""
1. train lại Logistic Regression pipeline
2. lấy feature names sau preprocessing
3. lấy coefficients
4. sort feature theo coefficient
5. lưu ra reports/logistic_coefficients.csv
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.load_data import load_german_credit


def main():
    df = load_german_credit()

    X = df.drop(columns=["target_original", "bad_risk"])
    y = df["bad_risk"]

    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

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

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": abs(coefficients),
    }).sort_values("coefficient", ascending=False)

    print("\nTop features increasing bad risk:")
    print(coef_df.head(15)[["feature", "coefficient"]])

    print("\nTop features decreasing bad risk:")
    print(coef_df.tail(15)[["feature", "coefficient"]])

    coef_df.sort_values("abs_coefficient", ascending=False).to_csv(
        "reports/logistic_coefficients.csv",
        index=False,
    )


if __name__ == "__main__":
    main()