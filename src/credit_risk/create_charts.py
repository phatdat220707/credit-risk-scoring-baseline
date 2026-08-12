from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from credit_risk.load_data import load_german_credit

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FIGURES_DIR = Path("reports/figures")
THRESHOLD_RESULTS_PATH = Path("reports/threshold_tuning_results.csv")


def plot_target_distribution():
    df = load_german_credit()

    counts = df["bad_risk"].value_counts().sort_index()
    labels = ["Good Risk", "Bad Risk"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts.values)
    plt.title("Target Distribution")
    plt.ylabel("Number of Applicants")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()


def plot_threshold_tuning():
    results = pd.read_csv(THRESHOLD_RESULTS_PATH)

    plt.figure(figsize=(8, 5))
    plt.plot(results["threshold"], results["precision"], marker="o", label="Precision")
    plt.plot(results["threshold"], results["recall"], marker="o", label="Recall")
    plt.plot(results["threshold"], results["f1"], marker="o", label="F1")
    plt.title("Threshold Tuning Metrics")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_metrics.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results["threshold"], results["business_cost"], marker="o")
    plt.title("Business Cost by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Business Cost")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_business_cost.png", dpi=150)
    plt.close()

def plot_confusion_matrix():
    df = load_german_credit()

    X = df.drop(columns=["target_original", "bad_risk"])
    y = df["bad_risk"]

    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.45).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Good Risk", "Bad Risk"],
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title("Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_target_distribution()
    plot_threshold_tuning()

    print(f"Saved charts to {FIGURES_DIR}")

    plot_confusion_matrix()

if __name__ == "__main__":
    main()