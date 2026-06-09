# ConvNeXt-Tiny Clean Split Experiment

Data root: `C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split`
Image size: `300`; resize: `340`
Best head epoch: `3`; best head selection score: `0.53054`
Best full epoch: `12`; best full selection score: `0.61437`

## Test Metrics

Single-view accuracy: 48.15%
Single-view macro F1: 0.4735
TTA accuracy: 48.15%
TTA macro F1: 0.4727
TTA adjacent accuracy: 86.57%
TTA MAE: 0.7130

## TTA Classification Report

```text
              precision    recall  f1-score   support

      Type 1     0.7000    1.0000    0.8235        14
      Type 2     0.9167    0.4583    0.6111        24
      Type 3     0.3636    0.1739    0.2353        23
      Type 4     0.4932    0.7200    0.5854        50
      Type 5     0.4062    0.3095    0.3514        42
      Type 6     0.3860    0.5000    0.4356        44
      Type 7     0.3636    0.2105    0.2667        19

    accuracy                         0.4815       216
   macro avg     0.5185    0.4818    0.4727       216
weighted avg     0.4897    0.4815    0.4624       216

```

## TTA Confusion Matrix

```text
[[14  0  0  0  0  0  0]
 [ 3 11  0  5  5  0  0]
 [ 1  1  4 17  0  0  0]
 [ 0  0  3 36  7  4  0]
 [ 0  0  4  7 13 18  0]
 [ 0  0  0  8  7 22  7]
 [ 2  0  0  0  0 13  4]]
```