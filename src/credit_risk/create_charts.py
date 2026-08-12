from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from credit_risk.load_data import load_german_credit


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


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_target_distribution()
    plot_threshold_tuning()

    print(f"Saved charts to {FIGURES_DIR}")


if __name__ == "__main__":
    main()