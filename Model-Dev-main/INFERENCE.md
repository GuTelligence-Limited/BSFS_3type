# BSFS Product Inference

This repository now keeps the product-facing inference path:

```text
7-class ConvNeXt-Tiny probabilities
-> 3-class product schema
-> Type 7 probability retained as a continuous watery-risk score
```

## Mode

- `product_3class`: primary product mode.
- `best_accuracy`: backward-compatible alias for `product_3class`.

## Command

```powershell
.\.venv\Scripts\python.exe bsfs_inference.py --image "path\to\image.jpg"
.\.venv\Scripts\python.exe bsfs_inference.py --image "path\to\image.jpg" --mode product_3class
```

Optional JSON output:

```powershell
.\.venv\Scripts\python.exe bsfs_inference.py --image "path\to\image.jpg" --output inference_result.json
```

## Product Schema

The recommended API payload is `product_schema`:

```json
{
  "schema_version": "bsfs_product_3class_v1",
  "primary_group": "Type 3/4 normal-range",
  "primary_group_confidence": 0.72,
  "primary_group_rule": "raw_top1_label_mapped_to_3class_product_group",
  "group_probabilities": {
    "Type 1/2 hard": 0.08,
    "Type 3/4 normal-range": 0.72,
    "Type 5/6/7 loose-watery": 0.20
  },
  "probability_argmax_group": "Type 3/4 normal-range",
  "bsfs_continuous_score": 4.12,
  "product_group_score": 3.80,
  "type7_probability": 0.04,
  "type7_risk": {
    "score": 0.04,
    "threshold_status": "not_calibrated_for_fixed_flag"
  },
  "raw_top1_label": "Type 4",
  "raw_top1_confidence": 0.51,
  "raw_probabilities": {
    "Type 1": 0.01,
    "Type 2": 0.07,
    "Type 3": 0.21,
    "Type 4": 0.51,
    "Type 5": 0.13,
    "Type 6": 0.03,
    "Type 7": 0.04
  }
}
```

## Product Notes

- `primary_group` uses the raw 7-class top-1 label mapped into:
  - `Type 1/2 hard`
  - `Type 3/4 normal-range`
  - `Type 5/6/7 loose-watery`
- This rule reached `79.63%` accuracy on the current clean test set.
- `type7_probability` is a continuous risk score. Do not convert it to a fixed warning flag yet; validation-selected thresholds did not generalize to clean test.
- `probability_argmax_group` is exposed because summed group probabilities can disagree with raw-top1 group mapping.
