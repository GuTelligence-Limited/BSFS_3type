# 4-Class Grouped BSFS Metrics

Prediction CSV: `checkpoints_clean_split_convnext_tiny\all_predictions_tta.csv`
Total images: 216
Correct: 152
Grouped accuracy: 70.37%
Grouped macro F1: 0.6292
Mean grouped confidence: 0.8370
Mean grouped confidence for correct predictions: 0.8496

## Classification Report

```text
                       precision    recall  f1-score   support

        Type 1/2 hard     0.8750    0.7368    0.8000        38
Type 3/4 normal-range     0.7143    0.8219    0.7643        73
  Type 5/6 soft-loose     0.6742    0.6977    0.6857        86
        Type 7 watery     0.3636    0.2105    0.2667        19

             accuracy                         0.7037       216
            macro avg     0.6568    0.6167    0.6292       216
         weighted avg     0.6957    0.7037    0.6955       216

```

## Confusion Matrix

```text
[[28  5  5  0]
 [ 2 60 11  0]
 [ 0 19 60  7]
 [ 2  0 13  4]]
```