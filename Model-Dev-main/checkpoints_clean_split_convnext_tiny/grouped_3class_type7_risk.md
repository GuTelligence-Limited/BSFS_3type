# 3-Class BSFS Product Metrics and Type 7 Risk

Prediction CSV: `checkpoints_clean_split_convnext_tiny\all_predictions_tta.csv`
Total images: 216
3-class grouped accuracy: 79.63%
3-class grouped macro F1: 0.7946
3-class grouped weighted F1: 0.7974
Type 7 support: 19
Type 7 ROC-AUC: 0.7457
Type 7 average precision: 0.2306

## Type 7 Risk Thresholds

| Threshold | Precision | Recall | FNR | FPR | Flagged | TP | FP | FN | TN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.2667 | 0.4211 | 0.5789 | 0.1117 | 0.1389 | 8 | 22 | 11 | 175 |
| 0.20 | 0.3684 | 0.3684 | 0.6316 | 0.0609 | 0.0880 | 7 | 12 | 12 | 185 |
| 0.30 | 0.3571 | 0.2632 | 0.7368 | 0.0457 | 0.0648 | 5 | 9 | 14 | 188 |
| 0.40 | 0.3636 | 0.2105 | 0.7895 | 0.0355 | 0.0509 | 4 | 7 | 15 | 190 |
| 0.50 | 0.3636 | 0.2105 | 0.7895 | 0.0355 | 0.0509 | 4 | 7 | 15 | 190 |

## Classification Report

```text
                         precision    recall  f1-score   support

          Type 1/2 hard     0.8750    0.7368    0.8000        38
  Type 3/4 normal-range     0.7143    0.8219    0.7643        73
Type 5/6/7 loose-watery     0.8400    0.8000    0.8195       105

               accuracy                         0.7963       216
              macro avg     0.8098    0.7863    0.7946       216
           weighted avg     0.8037    0.7963    0.7974       216

```

## Confusion Matrix

```text
[[28  5  5]
 [ 2 60 11]
 [ 2 19 84]]
```