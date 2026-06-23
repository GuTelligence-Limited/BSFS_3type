# GuTelligence BSFS Product Inference

This repository is now scoped to the current product-facing BSFS inference path:

```text
stool image
-> ConvNeXt-Tiny 7-class probabilities
-> 3-class product schema
-> Type 7 probability as continuous watery-risk score
```

The intended deployment path is a hosted API, such as a Hugging Face Space or endpoint, consumed by the Android application.

## Current Product Schema

Primary output:

```text
Type 1/2 hard
Type 3/4 normal-range
Type 5/6/7 loose-watery
```

Type 7 is not discarded. It is exposed as:

```text
type7_probability
```

This remains a continuous risk score. A fixed Type 7 warning threshold is not enabled because validation-selected thresholds did not generalize to clean test.

## Key Files

```text
bsfs_inference.py
calibrate_type7_risk.py
evaluate_3class_type7_risk.py
model.py
model_registry.json
train_convnext_tiny_3class_type7_aux.py
INFERENCE.md
MODEL_DEVELOPMENT_PLAN.md
DEVELOPMENT_LOG.md
CTO_MODEL_PROGRESS_BRIEF.tex
```

Purpose:

- `bsfs_inference.py`: single-image inference entry point and product schema output.
- `model_registry.json`: active model registry, currently `product_3class`.
- `evaluate_3class_type7_risk.py`: 3-class grouped metrics and Type 7 risk metrics.
- `calibrate_type7_risk.py`: validation/test Type 7 threshold calibration experiment.
- `model.py`: legacy EfficientNet-B4 model class kept only for compatibility/reference.
- `train_convnext_tiny_3class_type7_aux.py`: experimental 3-class + Type 7 auxiliary training script, retained for future comparison.
- `INFERENCE.md`: API/output schema details.

## Active Model

```text
checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth
```

Backbone:

```text
ConvNeXt-Tiny
```

Current clean-test product metric:

```text
3-class grouped accuracy : 79.63%
3-class macro F1         : 0.7946
Type 7 ROC-AUC           : 0.6815
Type 7 AP                : 0.2214
```

## Inference

Run default product inference:

```powershell
.\.venv\Scripts\python.exe bsfs_inference.py --image "path\to\image.jpg"
```

Save JSON:

```powershell
.\.venv\Scripts\python.exe bsfs_inference.py --image "path\to\image.jpg" --output inference_result.json
```

The main output object is:

```text
product_schema
```

See `INFERENCE.md` for the JSON contract.

## Calibration

Regenerate Type 7 calibration reports:

```powershell
.\.venv\Scripts\python.exe calibrate_type7_risk.py --min-recall 0.50
```

Outputs:

```text
type7_risk_calibration/
```

## Model Visualization

The Hugging Face Space includes a `Model Architecture` tab that shows a
`torchinfo` summary for the ConvNeXt-Tiny product model:

```text
hf_space_mvp/app.py
```

For graph-level inspection in Netron, export the active model to ONNX:

```powershell
.\.venv\Scripts\python.exe export_model_for_netron.py
```

Default output:

```text
model_exports/bsfs_convnext_tiny_product.onnx
```

Open the ONNX file with Netron to inspect the model graph.

## Notes for Android Integration

The Android app should not run model logic directly at this stage. It should call a hosted API that returns `product_schema`.

Recommended API response fields for Android:

```text
primary_group
primary_group_confidence
bsfs_continuous_score
product_group_score
type7_probability
raw_top1_label
raw_probabilities
```

Do not hard-code a Type 7 threshold in Android yet. Display `type7_probability` as a continuous risk score until device-native calibration is available.
