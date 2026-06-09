# 3-Class BSFS Product Metrics and Type 7 Risk

Prediction CSV: `C:\gutelligence\cv_model\Model-Dev-main\type7_risk_calibration\valid_predictions_full_probs.csv`
Total images: 214
3-class grouped accuracy: 79.44%
3-class grouped macro F1: 0.7694
3-class grouped weighted F1: 0.7958
Type 7 support: 19
Type 7 ROC-AUC: 0.9649
Type 7 average precision: 0.6731

## Type 7 Risk Thresholds

| Threshold | Precision | Recall | FNR | FPR | Flagged | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.4286 | 0.9474 | 0.0526 | 0.1231 | 0.1963 | 18 | 24 | 1 | 171 |
| 0.20 | 0.5455 | 0.9474 | 0.0526 | 0.0769 | 0.1542 | 18 | 15 | 1 | 180 |
| 0.30 | 0.6667 | 0.8421 | 0.1579 | 0.0410 | 0.1121 | 16 | 8 | 3 | 187 |
| 0.40 | 0.7143 | 0.7895 | 0.2105 | 0.0308 | 0.0981 | 15 | 6 | 4 | 189 |
| 0.50 | 0.6875 | 0.5789 | 0.4211 | 0.0256 | 0.0748 | 11 | 5 | 8 | 190 |

## Classification Report

```text
                         precision    recall  f1-score   support

          Type 1/2 hard     0.6757    0.6757    0.6757        37
  Type 3/4 normal-range     0.7381    0.8493    0.7898        73
Type 5/6/7 loose-watery     0.8925    0.7981    0.8426       104

               accuracy                         0.7944       214
              macro avg     0.7687    0.7744    0.7694       214
           weighted avg     0.8023    0.7944    0.7958       214

```

## Confusion Matrix

```text
[[25  6  6]
 [ 7 62  4]
 [ 5 16 83]]
```