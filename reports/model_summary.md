# Credit Risk Model Summary

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

- Threshold: 0.45
- Precision: 0.521
- Recall: 0.833
- F1: 0.641
- FP: 46
- FN: 10
- Business cost: 96

## Top Features Increasing Bad Risk

| feature                           |   coefficient |
|:----------------------------------|--------------:|
| cat__purpose_A46                  |      0.958526 |
| cat__property_A124                |      0.63535  |
| cat__checking_account_status_A11  |      0.620797 |
| cat__credit_history_A30           |      0.589948 |
| cat__savings_account_A61          |      0.584578 |
| cat__purpose_A45                  |      0.573417 |
| cat__foreign_worker_A201          |      0.568303 |
| cat__purpose_A40                  |      0.488497 |
| cat__housing_A151                 |      0.460398 |
| cat__other_installment_plans_A141 |      0.428027 |

## Top Features Decreasing Bad Risk

| feature                          |   coefficient |
|:---------------------------------|--------------:|
| cat__checking_account_status_A14 |     -0.98741  |
| cat__purpose_A41                 |     -0.88708  |
| cat__credit_history_A34          |     -0.852003 |
| cat__savings_account_A64         |     -0.676792 |
| cat__foreign_worker_A202         |     -0.651421 |
| cat__employment_since_A74        |     -0.595362 |
| cat__purpose_A410                |     -0.58943  |
| cat__personal_status_sex_A93     |     -0.535998 |
| cat__housing_A153                |     -0.50585  |
| cat__other_debtors_A103          |     -0.452236 |

## Notes

Positive coefficients increase the predicted probability of `bad_risk = 1`.
Negative coefficients decrease the predicted probability of `bad_risk = 1`.

These coefficients describe associations learned by the fitted Logistic Regression model. They should not be interpreted as causal effects.
