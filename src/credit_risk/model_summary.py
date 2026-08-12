from pathlib import Path

import pandas as pd


REPORTS_DIR = Path("reports")
THRESHOLD_PATH = REPORTS_DIR / "threshold_tuning_results.csv"
COEFFICIENTS_PATH = REPORTS_DIR / "logistic_coefficients.csv"
SUMMARY_PATH = REPORTS_DIR / "model_summary.md"


def main():
    threshold_results = pd.read_csv(THRESHOLD_PATH)
    coefficients = pd.read_csv(COEFFICIENTS_PATH)

    best_threshold = threshold_results.sort_values("business_cost").iloc[0]

    top_increasing = coefficients.sort_values("coefficient", ascending=False).head(10)
    top_decreasing = coefficients.sort_values("coefficient", ascending=True).head(10)

    summary = f"""# Credit Risk Model Summary

## Dataset

- Dataset: UCI Statlog German Credit Data
- Samples: 1000
- Features: 20
- Target: `bad_risk`
- Mapping:
  - `0`: good credit risk
  - `1`: bad credit risk

## Business Cost

The dataset provides a cost matrix where predicting a bad customer as good is more expensive than predicting a good customer as bad.

- False Positive cost: 1
- False Negative cost: 5

## Best Threshold By Business Cost

- Threshold: {best_threshold["threshold"]:.2f}
- Precision: {best_threshold["precision"]:.3f}
- Recall: {best_threshold["recall"]:.3f}
- F1: {best_threshold["f1"]:.3f}
- FP: {int(best_threshold["fp"])}
- FN: {int(best_threshold["fn"])}
- Business cost: {int(best_threshold["business_cost"])}

## Top Features Increasing Bad Risk

{top_increasing[["feature", "coefficient"]].to_markdown(index=False)}

## Top Features Decreasing Bad Risk

{top_decreasing[["feature", "coefficient"]].to_markdown(index=False)}

## Notes

Positive coefficients increase the predicted probability of `bad_risk = 1`.
Negative coefficients decrease the predicted probability of `bad_risk = 1`.

These coefficients describe associations learned by the fitted Logistic Regression model. They should not be interpreted as causal effects.
"""

    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()