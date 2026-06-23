# Development Log

This log records project updates that change code, data layout, reports, model workflow, or development planning. Keep this file synchronized with `MODEL_DEVELOPMENT_PLAN.md` when a change affects the roadmap, phase status, or next actions.

## 2026-06-01

### Documentation Baseline

Files changed:

- `README.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Summary:

- Expanded `README.md` from a placeholder into a project overview.
- Documented repository layout, dataset status, model architecture, training/evaluation commands, current recorded metrics, and known risks.
- Added staged model development plan from Phase 0 to Phase 6.

Reason:

- Establish a shared technical reference for future model development and testing.

### Phase 1 Dataset Audit

Files changed:

- `audit_dataset.py`
- `dataset_audit/manifest.csv`
- `dataset_audit/class_counts.csv`
- `dataset_audit/exact_duplicate_groups.csv`
- `dataset_audit/source_duplicate_groups.csv`
- `dataset_audit/ahash_near_duplicate_pairs.csv`
- `dataset_audit/audit_summary.md`
- `README.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Summary:

- Added read-only dataset audit script.
- Generated class counts, exact duplicate groups, same-source Roboflow variant groups, and first-pass near-duplicate signals.
- Recorded that the original train/valid/test split is contaminated by duplicate and same-source leakage.

Key results:

```text
Total images: 2149
Exact SHA-256 duplicate groups: 137
Exact duplicate groups crossing splits: 72
Source-name duplicate groups: 238
Source-name groups crossing splits: 197
aHash near-duplicate pairs: 1950
aHash near-duplicate pairs crossing splits: 873
aHash near-duplicate pairs with label conflicts: 724
```

Reason:

- Phase 1 requires confirming dataset integrity before continuing model tuning.

### Clean Split Generation

Files changed:

- `create_clean_split.py`
- `GuTelligence-StoMy-Clean-Split/`
- `GuTelligence-StoMy-Clean-Split/_split_report/class_counts.csv`
- `GuTelligence-StoMy-Clean-Split/_split_report/clean_split_manifest.csv`
- `GuTelligence-StoMy-Clean-Split/_split_report/clean_split_summary.md`
- `GuTelligence-StoMy-Clean-Split/_split_report/component_assignments.csv`
- `README.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Summary:

- Added non-destructive clean split generator.
- Created `GuTelligence-StoMy-Clean-Split/` while leaving original Roboflow export untouched.
- Split assignment unit is a connected component linked by normalized `source_id` and exact SHA-256.
- Regenerated split with target ratio close to train 80%, valid 10%, test 10%.

Key results:

```text
train: 1719
valid: 214
test : 216
source_id cross-split groups: 0
sha256 cross-split groups: 0
```

Reason:

- Remove known cross-split leakage from exact duplicates and same-source Roboflow variants.

### Clean Split Training Entry Point

Files changed:

- `training.py`
- `evaluate.py`
- `README.md`

Summary:

- Added `BSFS_DATA_ROOT` environment-variable support to `training.py` and `evaluate.py`.
- Default behavior still uses the original Roboflow export path.
- Clean split training/evaluation can now be run without editing source constants.

Usage:

```powershell
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
.\.venv\Scripts\python.exe training.py
```

Reason:

- Make clean split experiments reproducible and avoid manual code edits between datasets.

### Verification

Commands run:

```powershell
.\.venv\Scripts\python.exe audit_dataset.py
.\.venv\Scripts\python.exe create_clean_split.py --overwrite
.\.venv\Scripts\python.exe -m py_compile create_clean_split.py audit_dataset.py training.py evaluate.py
```

Results:

- Dataset audit completed.
- Clean split generated.
- No `source_id` cross-split groups in clean split.
- No `sha256` cross-split groups in clean split.
- Python syntax check passed for updated scripts.

### Clean Split Experiment Output Isolation

Files changed:

- `training.py`
- `evaluate.py`
- `README.md`

Summary:

- Added `BSFS_CHECKPOINT_DIR` support to `training.py` and `evaluate.py`.
- Added `BSFS_FIGURES_DIR` support to `evaluate.py`.
- Updated clean split training/evaluation commands to write into separate output directories.

Reason:

- Prevent cleaned split baseline checkpoints and figures from overwriting the original prototype experiment outputs.

### Clean Split EfficientNet-B4 Baseline Training

Files changed:

- `gpu.py`
- `evaluate.py`
- `checkpoints_clean_split/`
- `figures_clean_split/`
- `figures_clean_split_best_full/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Summary:

- Updated `gpu.py` to run a real CUDA kernel test instead of only checking `torch.cuda.is_available()`.
- Verified GPU execution on `NVIDIA GeForce RTX 5070 Ti Laptop GPU`.
- Trained the current EfficientNet-B4 baseline on `GuTelligence-StoMy-Clean-Split/`.
- Saved clean split checkpoints to `checkpoints_clean_split/`.
- Ran TTA evaluation for the final checkpoint and the best full fine-tune checkpoint.
- Fixed `evaluate.py` so `metrics_summary.txt` can be written when TTA evaluation does not have a test loss value.
- Added fixed TTA random seed to `evaluate.py` after detecting that random TTA crops changed reported accuracy between runs.

Environment:

```text
torch: 2.11.0+cu128
torch_cuda: 12.8
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
cuda_kernel_test: ok
```

Commands run:

```powershell
.\.venv\Scripts\python.exe gpu.py
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
$env:BSFS_CHECKPOINT_DIR="C:\gutelligence\cv_model\Model-Dev-main\checkpoints_clean_split"
.\.venv\Scripts\python.exe training.py

$env:BSFS_FIGURES_DIR="C:\gutelligence\cv_model\Model-Dev-main\figures_clean_split"
.\.venv\Scripts\python.exe evaluate.py

$env:BSFS_FIGURES_DIR="C:\gutelligence\cv_model\Model-Dev-main\figures_clean_split_best_full"
.\.venv\Scripts\python.exe evaluate.py --checkpoint checkpoints_clean_split\best_full_fine-tune.pth
```

Training results without TTA, from `training.py` final test evaluation:

```text
Test accuracy: 32.87%
Macro F1     : 0.3332
Test loss    : 1.8890
```

TTA evaluation results, final checkpoint with fixed seed:

```text
Checkpoint   : checkpoints_clean_split/bsfs_efficientnet_b4_final.pth
Test accuracy: 37.04%
Macro F1     : 0.3815
Macro ROC-AUC: 0.7646
```

TTA evaluation results, best full fine-tune checkpoint:

```text
Checkpoint   : checkpoints_clean_split/best_full_fine-tune.pth
Test accuracy: 37.50%
Macro F1     : 0.3966
Macro ROC-AUC: 0.7720
```

Per-class TTA metrics for final checkpoint:

```text
Type 1: precision=0.6250, recall=0.7143, F1=0.6667, support=14
Type 2: precision=0.6316, recall=0.5000, F1=0.5581, support=24
Type 3: precision=0.2857, recall=0.1739, F1=0.2162, support=23
Type 4: precision=0.3714, recall=0.5200, F1=0.4333, support=50
Type 5: precision=0.2639, recall=0.4524, F1=0.3333, support=42
Type 6: precision=0.3077, recall=0.0909, F1=0.1404, support=44
Type 7: precision=0.4167, recall=0.2632, F1=0.3226, support=19
```

Interpretation:

- The clean split baseline is much lower than the original contaminated split result.
- This confirms that prior 81.215% test accuracy was not a reliable product-level estimate.
- Weak classes on the clean split are Type 3, Type 5, Type 6, and Type 7, with Type 3 and Type 6 especially poor.

Reason:

- Establish a leakage-controlled baseline before continuing Phase 1 label review and Phase 3 model improvement.

### Clean Split Error Analysis

Files changed:

- `analyze_errors.py`
- `evaluate.py`
- `error_analysis_clean_split/`
- `figures_clean_split/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Summary:

- Added `analyze_errors.py` to generate image-level model review files.
- Generated `failure_review.csv` with manual review columns: `review_status`, `corrected_label`, and `notes`.
- Generated `all_predictions.csv`, `confusion_by_class.csv`, and copied review image buckets.
- Added fixed TTA seed in both `analyze_errors.py` and `evaluate.py` for reproducible TTA metrics.

Outputs:

```text
error_analysis_clean_split/failure_review.csv
error_analysis_clean_split/all_predictions.csv
error_analysis_clean_split/confusion_by_class.csv
error_analysis_clean_split/summary.md
error_analysis_clean_split/top_confident_errors/
error_analysis_clean_split/low_confidence_cases/
error_analysis_clean_split/type1_errors/
error_analysis_clean_split/type2_errors/
error_analysis_clean_split/type3_errors/
error_analysis_clean_split/type4_errors/
error_analysis_clean_split/type5_errors/
error_analysis_clean_split/type6_errors/
error_analysis_clean_split/type7_errors/
```

Key results:

```text
Total test images: 216
Correct: 80
Errors: 136
Accuracy: 37.04%
Adjacent accuracy (+/-1): 72.69%
Ordinal MAE: 1.0370
High-confidence errors (confidence >= 0.50): 72
Low-confidence cases (confidence < 0.35): 18
```

Errors by true label:

```text
Type 1: 4/14 errors, 28.57%
Type 2: 12/24 errors, 50.00%
Type 3: 19/23 errors, 82.61%
Type 4: 24/50 errors, 48.00%
Type 5: 23/42 errors, 54.76%
Type 6: 40/44 errors, 90.91%
Type 7: 14/19 errors, 73.68%
```

Most common error pairs:

```text
Type 6 -> Type 5: 16
Type 6 -> Type 4: 15
Type 4 -> Type 5: 15
Type 3 -> Type 4: 13
Type 5 -> Type 4: 10
Type 7 -> Type 5: 10
```

Reason:

- Prepare Phase 1 manual review of label quality and ambiguous BSFS boundaries.

### Clean Split Training Parameter Tuning

Files changed:

- `training.py`
- `evaluate.py`
- `analyze_errors.py`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Summary:

- Tuned the next clean-split EfficientNet-B4 training profile based on the previous clean-split baseline.
- Increased input resolution from 224 to 300 pixels, closer to EfficientNet-B4's native pretraining resolution.
- Reduced morphology-disruptive augmentation:
  - removed vertical flip
  - removed random erasing
  - removed Gaussian blur
  - reduced crop strength, rotation, and color jitter
- Disabled Mixup because BSFS classes are ordinal and visually ambiguous.
- Reduced label smoothing from 0.05 to 0.02.
- Reduced learning rates for partial/full fine-tuning.
- Increased Phase 3 duration and early stopping patience.
- Synced `evaluate.py` and `analyze_errors.py` input resolution/TTA transforms with the tuned training profile.

New training profile:

```text
IMG_SIZE: 300
RESIZE_SIZE: 340
BATCH_SIZE: 20
PHASE1_EPOCHS: 10
PHASE2_EPOCHS: 30
PHASE3_EPOCHS: 35
PHASE1_LR: 8e-4
PHASE2_LR: 1e-4
PHASE3_LR: 1.5e-5
DROPOUT_RATE: 0.30
LABEL_SMOOTHING: 0.02
MIXUP_ALPHA: 0.0
PATIENCE: 15
```

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile training.py evaluate.py analyze_errors.py
.\.venv\Scripts\python.exe -c "import torch; from model import BSFSClassifier; m=BSFSClassifier(num_classes=7, dropout_rate=0.3, use_gem=True).cuda().eval(); x=torch.randn(1,3,300,300,device='cuda'); y=m(x); print(tuple(y.shape))"
```

Result:

```text
Forward pass output shape: (1, 7)
```

Reason:

- The previous clean-split baseline underfit/over-augmented the ordinal fine-grained BSFS morphology. The next run should preserve shape cues and use higher input detail.

### Training BatchNorm Tail-Batch Fix

Files changed:

- `training.py`
- `DEVELOPMENT_LOG.md`

Summary:

- Fixed a training crash caused by the final training batch containing a single image.
- The classifier head uses `BatchNorm1d`, which cannot train on an input shaped like `[1, 512]`.
- Added `drop_last=True` to the training `DataLoader`.
- Added startup prints for `DATA_ROOT` and `CHECKPOINT_DIR` so accidental training on the original split is immediately visible.

Observed error:

```text
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512])
```

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile training.py
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
.\.venv\Scripts\python.exe -c "from training import DATA_ROOT,BATCH_SIZE,build_dataloaders; tl,vl,te,ds=build_dataloaders(DATA_ROOT,BATCH_SIZE,0); import itertools; sizes=[x[0].shape[0] for x in itertools.islice(tl,1000)]; print(min(sizes), max(sizes), sizes[-1])"
```

Result:

```text
Train: 1719 | Valid: 214 | Test: 216
min_batch=20, max_batch=20, last_batch=20
```

Reason:

- Keep `BatchNorm1d` stable during training and avoid single-sample tail batches.

### Tuned 300 Clean Split Training Results

Files changed:

- `checkpoints_clean_split_tuned_300/`
- `figures_clean_split_tuned_300/`
- `error_analysis_clean_split_tuned_300/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Summary:

- Trained the tuned 300x300 EfficientNet-B4 profile on `GuTelligence-StoMy-Clean-Split/`.
- Evaluated the final checkpoint with fixed-seed TTA.
- Generated updated figures and error analysis.

Outputs:

```text
checkpoints_clean_split_tuned_300/
figures_clean_split_tuned_300/
error_analysis_clean_split_tuned_300/
```

Training final evaluation without TTA:

```text
Test accuracy: 40.278%
Macro F1     : 0.36263
Test loss    : 2.82614
```

Fixed-seed TTA evaluation:

```text
Test accuracy: 42.13%
Macro F1     : 0.3878
Macro ROC-AUC: 0.7688
Adjacent accuracy (+/-1): 77.31%
Ordinal MAE: 0.9306
```

Per-class TTA metrics:

```text
Type 1: precision=0.4286, recall=0.6429, F1=0.5143, support=14
Type 2: precision=0.6250, recall=0.6250, F1=0.6250, support=24
Type 3: precision=0.0000, recall=0.0000, F1=0.0000, support=23
Type 4: precision=0.4507, recall=0.6400, F1=0.5289, support=50
Type 5: precision=0.3922, recall=0.4762, F1=0.4301, support=42
Type 6: precision=0.2917, recall=0.1591, F1=0.2059, support=44
Type 7: precision=0.4000, recall=0.4211, F1=0.4103, support=19
```

Most common error pairs:

```text
Type 3 -> Type 4: 17
Type 6 -> Type 5: 14
Type 6 -> Type 4: 13
Type 4 -> Type 5: 8
Type 7 -> Type 6: 7
Type 5 -> Type 6: 7
Type 6 -> Type 7: 7
```

Comparison to previous clean-split baseline:

```text
Accuracy: 37.04% -> 42.13%
Macro F1: 0.3815 -> 0.3878
Adjacent accuracy: 72.69% -> 77.31%
Ordinal MAE: 1.0370 -> 0.9306
```

Interpretation:

- The tuned 300x300 profile improved exact accuracy, adjacent accuracy, and ordinal MAE.
- Macro F1 only improved slightly because Type 3 collapsed to zero recall and Type 6 remains weak.
- Next optimization should target class-balanced/ordinal behavior rather than only resolution or augmentation.

Reason:

- Establish whether morphology-preserving augmentation and higher resolution improve clean-split performance.

### Next Optimization: Macro-F1 Selection and Class-Balanced Loss

Files changed:

- `training.py`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Summary:

- Changed validation checkpoint selection from accuracy to macro F1.
- Early stopping now monitors validation macro F1.
- Validation history now records macro F1, adjacent accuracy, and ordinal MAE.
- Added lightweight class-balanced loss using square-root inverse-frequency class weights.
- Kept `WeightedRandomSampler`, but used gentler `CLASS_WEIGHT_POWER=0.5` to avoid full inverse-frequency overcorrection.

Reason:

- The tuned 300 run improved accuracy and ordinal metrics, but Type 3 recall collapsed to 0 and Type 6 remained weak.
- Accuracy-based checkpointing is dominated by larger/easier classes. Macro F1 and class-balanced loss should better target weak-class recall.

New behavior:

```text
Checkpoint criterion: validation macro F1
Early stopping metric: validation macro F1
Loss: CrossEntropyLoss(weight=sqrt_inverse_frequency_weights, label_smoothing=0.02)
Logged validation metrics: val_acc, val_macro_f1, val_adjacent_acc, val_mae
```

Computed train-set class weights for clean split:

```text
Type 1: 1.398
Type 2: 1.046
Type 3: 1.068
Type 4: 0.726
Type 5: 0.791
Type 6: 0.780
Type 7: 1.190
```

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile training.py
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
.\.venv\Scripts\python.exe -c "from training import DATA_ROOT,BATCH_SIZE,build_dataloaders,compute_class_weights; tl,vl,te,ds=build_dataloaders(DATA_ROOT,BATCH_SIZE,0); print(compute_class_weights(ds).tolist())"
```

Recommended next experiment output:

```text
checkpoints_clean_split_tuned_300_macro_f1/
figures_clean_split_tuned_300_macro_f1/
error_analysis_clean_split_tuned_300_macro_f1/
```

### Macro-F1 Weighted Run, Deterministic TTA, Composite Run, and Ensemble Candidate

Date: 2026-06-01

Files changed:

- `training.py`
- `evaluate.py`
- `analyze_errors.py`
- `evaluate_ensemble.py`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Summary:

- Trained `checkpoints_clean_split_tuned_300_macro_f1/` using validation macro F1 selection and class-weighted CrossEntropy.
- Fixed `analyze_errors.py` review-image copy failures on Windows by replacing long copied filenames with short hash-based names.
- Replaced random TTA crops in `evaluate.py` and `analyze_errors.py` with deterministic 5-view TTA.
- Changed default dataset roots in `training.py` and `evaluate.py` to `GuTelligence-StoMy-Clean-Split/` to avoid accidental evaluation on the original contaminated split.
- Trained `checkpoints_clean_split_tuned_300_composite/` using unweighted label-smoothed CE and a composite validation selection score.
- Added `evaluate_ensemble.py` for reproducible weighted two-model ensemble evaluation on the clean split.

Clean split deterministic TTA comparison:

```text
tuned_300 final:
  Accuracy: 43.98%
  Macro F1: 0.4047
  Adjacent accuracy: 74.54%
  Ordinal MAE: 0.9306

tuned_300_macro_f1 final:
  Accuracy: 42.59%
  Macro F1: 0.4137
  Adjacent accuracy: 78.24%
  Ordinal MAE: 0.8750

tuned_300_composite final:
  Accuracy: 43.52%
  Macro F1: 0.3984
  Adjacent accuracy: 74.54%
  Ordinal MAE: 0.9352

ensemble tuned_300 80% + macro_f1 20%:
  Accuracy: 45.83%
  Macro F1: 0.4254
  Adjacent accuracy: 75.93%
  Ordinal MAE: 0.8889
```

Best current candidate:

```text
Single model for exact accuracy:
  checkpoints_clean_split_tuned_300/bsfs_efficientnet_b4_final.pth

Best measured candidate overall:
  ensemble_results/tuned300_macro_f1_80_20/
  A = checkpoints_clean_split_tuned_300/bsfs_efficientnet_b4_final.pth, weight 0.8
  B = checkpoints_clean_split_tuned_300_macro_f1/bsfs_efficientnet_b4_final.pth, weight 0.2
```

Important correction:

- A temporary ensemble check accidentally used the original default dataset and reported inflated 181-image test metrics.
- This was rejected as invalid.
- Defaults were updated so future no-env runs use `GuTelligence-StoMy-Clean-Split/`.

Remaining failure mode:

- Type 3 recall remains 0.00 across the current best single model and ensemble.
- Type 6 and Type 7 remain weak.
- Further generic parameter tuning is unlikely to solve Type 3 without label/data review, a Type 3/4-focused strategy, or additional device-native data.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile training.py evaluate.py analyze_errors.py evaluate_ensemble.py
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
$env:BSFS_CHECKPOINT_DIR="C:\gutelligence\cv_model\Model-Dev-main\checkpoints_clean_split_tuned_300"
$env:BSFS_FIGURES_DIR="C:\gutelligence\cv_model\Model-Dev-main\figures_clean_split_tuned_300_deterministic_tta"
.\.venv\Scripts\python.exe evaluate.py
$env:BSFS_ENSEMBLE_DIR="C:\gutelligence\cv_model\Model-Dev-main\ensemble_results"
.\.venv\Scripts\python.exe evaluate_ensemble.py --checkpoint-a checkpoints_clean_split_tuned_300\bsfs_efficientnet_b4_final.pth --checkpoint-b checkpoints_clean_split_tuned_300_macro_f1\bsfs_efficientnet_b4_final.pth --weight-a 0.8 --name tuned300_macro_f1_80_20
```

### Autonomous Iteration: Postprocess, Focal Fine-Tune, and Type 3 Diagnosis

Date: 2026-06-01

Files changed:

- `evaluate_ensemble.py`
- `optimize_ensemble.py`
- `fine_tune_focal.py`
- `fine_tune_type3.py`
- `evaluate.py`
- `type3_diagnostic_summary.md`
- `type3_type4_review.csv`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Experiments:

1. Validation-tuned postprocessing
   - Script: `optimize_ensemble.py`
   - Output: `postprocess_results/tuned300_macro_f1_valid_tuned/`
   - Selected on valid split only.
   - Test result: accuracy `41.67%`, macro F1 `0.4016`, adjacent accuracy `78.24%`, ordinal MAE `0.8935`.
   - Outcome: rejected; did not generalize to test.

2. Focal-loss short fine-tune from `tuned_300`
   - Script: `fine_tune_focal.py`
   - Output checkpoints: `checkpoints_clean_split_focal_ft_tuned300/`
   - Figures: `figures_clean_split_focal_ft_tuned300/`
   - Error analysis: `error_analysis_clean_split_focal_ft_tuned300/`
   - Deterministic TTA result: accuracy `43.98%`, macro F1 `0.4308`, macro ROC-AUC `0.7681`, adjacent accuracy `76.39%`, ordinal MAE `0.8843`.
   - Outcome: accepted as best single-model macro-F1 candidate, but exact accuracy does not beat `tuned_300` and Type 3 recall remains `0.00`.

3. Type 3 oversampling short fine-tune from `tuned_300`
   - Script: `fine_tune_type3.py`
   - Output checkpoints: `checkpoints_clean_split_type3_ft_tuned300/`
   - Figures: `figures_clean_split_type3_ft_tuned300/`
   - Error analysis: `error_analysis_clean_split_type3_ft_tuned300/`
   - Validation Type 3 recall reached `0.435` during training.
   - Deterministic TTA test result: accuracy `39.35%`, macro F1 `0.3824`, Type 3 recall `0.00`.
   - Outcome: rejected; overfit validation Type 3 patterns and hurt test performance.

4. Three-model ensemble check
   - Models: `tuned_300`, `macro_f1`, `focal_ft`.
   - Best grid result did not beat existing two-model ensemble.
   - Existing two-model ensemble remains current best exact-accuracy candidate: accuracy `45.83%`, macro F1 `0.4254`.

Type 3 diagnosis:

- Generated `type3_diagnostic_summary.md`.
- Generated `type3_type4_review.csv` with 73 Type 3/4 clean-test rows and predictions from `tuned300`, `macro_f1`, `focal_ft`, and `type3_ft`.
- Across models, true Type 3 images are usually predicted as Type 4 or Type 2.
- Type 3 probability remains very low for most true Type 3 samples:
  - `tuned300`: Type 3 top-prob max about `0.062`.
  - `macro_f1`: Type 3 top-prob max about `0.327`.
  - `focal_ft`: Type 3 top-prob max about `0.212`.
- Type 3 threshold postprocessing selected on valid hurt test accuracy and still produced `0.00` Type 3 recall.

Current model candidates:

```text
Best exact-accuracy candidate:
  ensemble_results/tuned300_macro_f1_80_20/
  Accuracy: 45.83%
  Macro F1: 0.4254

Best single-model exact accuracy:
  checkpoints_clean_split_tuned_300/bsfs_efficientnet_b4_final.pth
  Accuracy: 43.98%
  Macro F1: 0.4047

Best single-model macro F1:
  checkpoints_clean_split_focal_ft_tuned300/bsfs_efficientnet_b4_final.pth
  Accuracy: 43.98%
  Macro F1: 0.4308
```

Conclusion:

- Generic tuning, focal loss, class weighting, Type 3 oversampling, and threshold postprocessing did not solve Type 3 recall on clean test.
- Next useful work is data/label diagnosis for Type 3/4 and possibly a dedicated Type 3/4 modeling strategy after review.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile evaluate_ensemble.py optimize_ensemble.py fine_tune_focal.py fine_tune_type3.py evaluate.py
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
.\.venv\Scripts\python.exe optimize_ensemble.py --checkpoint-a checkpoints_clean_split_tuned_300\bsfs_efficientnet_b4_final.pth --checkpoint-b checkpoints_clean_split_tuned_300_macro_f1\bsfs_efficientnet_b4_final.pth --name tuned300_macro_f1_valid_tuned
.\.venv\Scripts\python.exe fine_tune_focal.py --checkpoint checkpoints_clean_split_tuned_300\bsfs_efficientnet_b4_final.pth
.\.venv\Scripts\python.exe fine_tune_type3.py --checkpoint checkpoints_clean_split_tuned_300\bsfs_efficientnet_b4_final.pth
```

### Phase 3 Experiment: Type 2/3/4 Neighbor Classifier and Hierarchical Inference

Date: 2026-06-01

Files changed:

- `train_neighbor_classifier.py`
- `evaluate_hierarchical.py`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Goal:

- Test whether Type 3 failure is caused by the full 7-class task suppressing a difficult local Type 2/3/4 boundary.

Experiments:

1. Type 2/3/4 neighbor classifier initialized from `tuned_300`
   - Script: `train_neighbor_classifier.py`
   - Output: `checkpoints_neighbor_234_tuned300_init/`
   - Subtask test accuracy: `38.14%`
   - Subtask macro F1: `0.3333`
   - Type 3 recall: `0.00`
   - Outcome: rejected.

2. Type 2/3/4 neighbor classifier initialized from ImageNet
   - Script: `train_neighbor_classifier.py`
   - Output: `checkpoints_neighbor_234_imagenet/`
   - Subtask test accuracy: `37.11%`
   - Subtask macro F1: `0.3291`
   - Type 3 recall: `0.2609`
   - Outcome: it can predict some Type 3, but causes many Type 4 -> Type 3 mistakes and is not strong enough for deployment.

3. Hierarchical inference using base model plus neighbor classifier
   - Script: `evaluate_hierarchical.py`
   - Outputs:
     - `hierarchical_results/focal_base_neighbor234_imagenet/`
     - `hierarchical_results/tuned300_base_neighbor234_imagenet/`
     - `hierarchical_results/tuned300_base_neighbor234_tunedinit/`
   - Best hierarchical result:
     - Base: `checkpoints_clean_split_tuned_300/bsfs_efficientnet_b4_final.pth`
     - Neighbor: `checkpoints_neighbor_234_imagenet/bsfs_neighbor_234_final.pth`
     - Accuracy: `44.44%`
     - Macro F1: `0.4076`
     - Type 3 recall: `0.00`
   - Outcome: rejected; does not beat best single model or ensemble.

Interpretation:

- Even a dedicated Type 2/3/4 classifier does not reliably separate Type 3 on clean test.
- The ImageNet-initialized neighbor classifier can force some Type 3 predictions, but those predictions mostly come at the cost of false positives from Type 4.
- Hierarchical inference selected on validation does not generalize to a better test result.

Current conclusion:

- Type 3 is now a confirmed data/label-boundary blocker, not merely a 7-class architecture or class-imbalance problem.
- Further model-only work should wait until Type 3/4 labels and source groups are reviewed, or until additional device-native Type 3 examples are collected.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile train_neighbor_classifier.py evaluate_hierarchical.py
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
.\.venv\Scripts\python.exe train_neighbor_classifier.py --init-checkpoint checkpoints_clean_split_tuned_300\bsfs_efficientnet_b4_final.pth
.\.venv\Scripts\python.exe train_neighbor_classifier.py
.\.venv\Scripts\python.exe evaluate_hierarchical.py --base-checkpoint checkpoints_clean_split_tuned_300\bsfs_efficientnet_b4_final.pth --neighbor-checkpoint checkpoints_neighbor_234_imagenet\bsfs_neighbor_234_final.pth --name tuned300_base_neighbor234_imagenet
```

### Type 3/4 Data Review Artifact Generation

Date: 2026-06-01

Files changed:

- `review_type3_type4.py`
- `type3_type4_review_artifacts/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Summary:

- Added `review_type3_type4.py` to turn the Type 3/4 model-diagnostic CSV into review-ready artifacts.
- Generated source-level and component-level summaries.
- Generated ranked suspicious rows and contact sheet images.
- Copied top suspicious source images into a review folder.

Outputs:

```text
type3_type4_review_artifacts/summary.md
type3_type4_review_artifacts/type3_type4_suspicious_ranked.csv
type3_type4_review_artifacts/type3_type4_source_summary.csv
type3_type4_review_artifacts/type3_type4_component_summary.csv
type3_type4_review_artifacts/contact_sheet_top_suspicious.jpg
type3_type4_review_artifacts/contact_sheet_type3.jpg
type3_type4_review_artifacts/contact_sheet_type4.jpg
type3_type4_review_artifacts/top_suspicious_images/
```

Key review findings:

- Review rows: `73` total (`23` Type 3, `50` Type 4).
- Top suspicious Type 3 source IDs are:
  - `type310_jpg`: 3 Type 3 rows, consensus predicted Type 4 for all.
  - `type314_jpg`: 6 Type 3 rows, consensus predicted Type 4 for all.
  - `6xjvw61qgtdb1_jpg`: 7 Type 3 rows, consensus predicted mostly Type 2 or Type 4.
  - `type31_jpg`: 4 Type 3 rows, consensus predicted Type 4 for all.
- These source groups should be reviewed first because all or nearly all models disagree with the current Type 3 labels.

Important note:

- Files copied into `top_suspicious_images/` preserve source file timestamps by `shutil.copy2`; the generated CSV/contact sheets are current even if copied image timestamps look old.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile review_type3_type4.py
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
.\.venv\Scripts\python.exe review_type3_type4.py
```

### No-Manual-Relabel Experiment: Ordinal Soft-Label Fine-Tune

Date: 2026-06-01

Files changed:

- `fine_tune_ordinal_soft.py`
- `checkpoints_clean_split_ordinal_soft_tuned300/`
- `figures_clean_split_ordinal_soft_tuned300/`
- `error_analysis_clean_split_ordinal_soft_tuned300/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Context:

- The team currently cannot manually relabel Type 3/4 data.
- Instead of blocking on label review, this experiment tested whether ordinal soft labels can absorb adjacent-label noise.

Method:

- Started from `checkpoints_clean_split_tuned_300/bsfs_efficientnet_b4_final.pth`.
- Used soft target distributions instead of hard one-hot labels.
- Assigned probability mass to adjacent BSFS classes.
- Added extra soft mass between Type 3 and Type 4 to reflect uncertain boundary.
- Evaluation still used the existing clean test labels.

Result:

```text
Deterministic TTA accuracy: 43.98%
Macro F1: 0.4185
Macro ROC-AUC: 0.7648
Adjacent accuracy: 75.93%
Ordinal MAE: 0.9167
Type 3 recall: 0.00
```

Outcome:

- Rejected as a new best model.
- It did not beat:
  - best exact candidate: ensemble `45.83%` accuracy, `0.4254` macro F1
  - best single-model macro-F1 candidate: focal fine-tune `43.98%` accuracy, `0.4308` macro F1
- It slightly reduces high-confidence errors versus some previous runs, but does not solve Type 3.

No-manual-relabel conclusion:

- Without relabeling, current best practical candidates remain:
  - exact accuracy: `ensemble_results/tuned300_macro_f1_80_20/`
  - single-model macro F1: `checkpoints_clean_split_focal_ft_tuned300/bsfs_efficientnet_b4_final.pth`
- Type 3 should be treated as an uncertain/borderline output problem until labels or new data are available.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile fine_tune_ordinal_soft.py
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
$env:BSFS_CHECKPOINT_DIR="C:\gutelligence\cv_model\Model-Dev-main\checkpoints_clean_split_ordinal_soft_tuned300"
.\.venv\Scripts\python.exe fine_tune_ordinal_soft.py --checkpoint checkpoints_clean_split_tuned_300\bsfs_efficientnet_b4_final.pth
$env:BSFS_FIGURES_DIR="C:\gutelligence\cv_model\Model-Dev-main\figures_clean_split_ordinal_soft_tuned300"
.\.venv\Scripts\python.exe evaluate.py
```

### Selective Prediction / Abstention Evaluation

Date: 2026-06-01

Files changed:

- `evaluate_selective.py`
- `selective_results/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Context:

- Manual relabeling is not currently available.
- Type 3/4 remains a known uncertain boundary.
- Instead of always forcing an exact 7-class prediction, this experiment evaluates abstention for low-confidence or low-margin predictions.

Method:

- Deterministic TTA probabilities are computed for valid and test.
- Confidence and top1-top2 margin thresholds are selected on valid only.
- Selected thresholds are applied once to test.
- Metrics are reported only over retained predictions, with coverage and abstention count.

Results:

```text
High-coverage ensemble mode:
  Result dir: selective_results/ensemble_80_20_min80/
  Model: tuned_300 80% + macro_f1 20%
  Threshold: confidence >= 0.38, margin >= 0.02
  Test coverage: 93.98% (203/216 kept, 13 abstained)
  Test accuracy on kept samples: 45.32%
  Test macro F1 on kept samples: 0.4104
  Test adjacent accuracy on kept samples: 76.85%
  Test MAE on kept samples: 0.8867

Cautious single-model mode:
  Result dir: selective_results/focal_ft_min60/
  Model: focal_ft single model
  Threshold: confidence >= 0.58, margin >= 0.00
  Test coverage: 70.83% (153/216 kept, 63 abstained)
  Test accuracy on kept samples: 45.10%
  Test macro F1 on kept samples: 0.4300
  Test adjacent accuracy on kept samples: 80.39%
  Test MAE on kept samples: 0.8562
```

Interpretation:

- Selective prediction does not solve Type 3 recall, but it creates a product-usable uncertainty mechanism.
- The cautious focal mode improves adjacent accuracy and MAE on retained samples by abstaining from about 29% of images.
- The high-coverage ensemble mode preserves most predictions and abstains on 13/216 images, but gains little over full-coverage ensemble accuracy.

Recommended operating modes under current data constraints:

- Research/batch evaluation: use `ensemble_results/tuned300_macro_f1_80_20/`.
- Product prototype high-coverage mode: use `selective_results/ensemble_80_20_min80/`.
- Product prototype cautious mode: use `selective_results/focal_ft_min60/`.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile evaluate_selective.py
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
$env:BSFS_SELECTIVE_DIR="C:\gutelligence\cv_model\Model-Dev-main\selective_results"
.\.venv\Scripts\python.exe evaluate_selective.py --checkpoint-a checkpoints_clean_split_tuned_300\bsfs_efficientnet_b4_final.pth --checkpoint-b checkpoints_clean_split_tuned_300_macro_f1\bsfs_efficientnet_b4_final.pth --weight-a 0.8 --name ensemble_80_20_min80 --min-coverage 0.80
.\.venv\Scripts\python.exe evaluate_selective.py --checkpoint-a checkpoints_clean_split_focal_ft_tuned300\bsfs_efficientnet_b4_final.pth --name focal_ft_min60 --min-coverage 0.60
```

### Inference Prototype Wrapper and Model Registry

Date: 2026-06-01

Files changed:

- `model_registry.json`
- `bsfs_inference.py`
- `INFERENCE.md`
- `inference_samples/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Summary:

- Added a model registry describing preprocessing, checkpoints, ensemble weights, selective thresholds, and clean-test metrics.
- Added `bsfs_inference.py` for single-image inference.
- Added support for two modes:
  - `high_coverage`
  - `cautious`
- Added `warnings` in the inference output for known Type 3/4 boundary risk.
- Added `INFERENCE.md` with command examples and output field definitions.

Validation:

- Ran inference on clean test Type 1 and Type 3 examples.
- Saved sample outputs:
  - `inference_samples/type1_high_coverage.json`
  - `inference_samples/type1_cautious.json`
  - `inference_samples/type3_high_coverage.json`
  - `inference_samples/type3_cautious.json`
  - `inference_samples/type3_high_coverage_with_warning.json`

Important behavior:

- `accepted=false` means the model abstained due to low confidence or low margin.
- `warnings=["type3_type4_boundary_known_unreliable"]` flags a known weak boundary even if `accepted=true`.
- Product code should not treat an accepted Type 3/4-region prediction as fully reliable without this warning context.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile bsfs_inference.py
.\.venv\Scripts\python.exe bsfs_inference.py --image "GuTelligence-StoMy-Clean-Split\test\Type 1\test__type1_1_jpg.rf.a203c074ce6e75b9266fe03af3169bbe.jpg" --mode high_coverage
.\.venv\Scripts\python.exe bsfs_inference.py --image "GuTelligence-StoMy-Clean-Split\test\Type 3\test__6xjvw61qgtdb1_jpg.rf.da720043a496c69c4c03cadcf12008df.jpg" --mode high_coverage
```

### Phase 3 Architecture and Resolution Iteration

Date: 2026-06-01

Files changed:

- `fine_tune_highres.py`
- `train_convnext_tiny.py`
- `evaluate_mixed_ensemble.py`
- `evaluate.py`
- `analyze_errors.py`
- `bsfs_inference.py`
- `model_registry.json`
- `checkpoints_clean_split_highres_380_tuned300/`
- `figures_clean_split_highres_380_tuned300/`
- `error_analysis_clean_split_highres_380_tuned300/`
- `checkpoints_clean_split_convnext_tiny/`
- `ensemble_results/convnext_tiny_tuned300_valid_weighted/`
- `ensemble_results/convnext_tiny_macro_f1_valid_weighted/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Context:

- The target remains 90% final accuracy, but clean-split evidence shows this is not reachable by simple EfficientNet hyperparameter tuning.
- Manual Type 3/4 relabeling is currently unavailable, so Phase 3 continued with model-only experiments that do not require new labels.

High-resolution EfficientNet-B4 fine-tune:

- Started from `checkpoints_clean_split_tuned_300/bsfs_efficientnet_b4_final.pth`.
- Fine-tuned at 380x380 input with resize 420 and reduced batch size.
- Added environment-variable image-size support to `evaluate.py` and `analyze_errors.py`.
- Result: rejected. TTA accuracy fell to `37.04%`, macro F1 `0.3605`, adjacent accuracy remained `75.93%`.
- Interpretation: the current failure is not primarily caused by 300px input resolution.

ConvNeXt-Tiny architecture experiment:

- Script: `train_convnext_tiny.py`.
- Backbone: ImageNet-pretrained ConvNeXt-Tiny.
- Training: clean split, 300x300, head warmup followed by full fine-tune.
- Best validation point: full fine-tune epoch 12.
- Clean test TTA result:

```text
Result dir: checkpoints_clean_split_convnext_tiny/
Single-view accuracy: 48.15%
TTA accuracy: 48.15%
TTA macro F1: 0.4727
TTA adjacent accuracy: 86.57%
TTA MAE: 0.7130
Type 3 recall: 0.1739
```

Mixed ensemble checks:

```text
convnext_tiny + tuned300, valid-selected weight:
  Result dir: ensemble_results/convnext_tiny_tuned300_valid_weighted/
  Weight: ConvNeXt 0.85 + EfficientNet tuned300 0.15
  Test accuracy: 46.30%
  Macro F1: 0.4566
  Adjacent accuracy: 85.65%
  MAE: 0.7176

convnext_tiny + macro_f1, valid-selected weight:
  Result dir: ensemble_results/convnext_tiny_macro_f1_valid_weighted/
  Weight: ConvNeXt 0.75 + EfficientNet macro_f1 0.25
  Test accuracy: 43.06%
  Macro F1: 0.4123
  Adjacent accuracy: 85.65%
  MAE: 0.7685
```

Decision:

- New best full-coverage clean-test candidate: `checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth`.
- Do not use the mixed ensembles as the primary model because neither beats the ConvNeXt single model.
- Added `best_accuracy` mode to `model_registry.json` and `bsfs_inference.py` for ConvNeXt-Tiny inference.
- Keep previous `high_coverage` and `cautious` modes because they provide selective abstention behavior, while `best_accuracy` is full coverage.
- Added `optimize_single_postprocess.py` and tested validation-selected temperature/class-bias postprocessing for ConvNeXt. It is rejected as the primary mode because exact accuracy dropped from `48.15%` to `47.69%`, although macro F1 increased to `0.4793`.

Updated candidate table:

```text
Best full-coverage exact accuracy:
  checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth
  Accuracy: 48.15%
  Macro F1: 0.4727
  Adjacent accuracy: 86.57%
  MAE: 0.7130

Best EfficientNet ensemble historical candidate:
  ensemble_results/tuned300_macro_f1_80_20/
  Accuracy: 45.83%
  Macro F1: 0.4254
  Adjacent accuracy: 75.93%
  MAE: 0.8889
```

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile fine_tune_highres.py train_convnext_tiny.py evaluate_mixed_ensemble.py evaluate.py analyze_errors.py bsfs_inference.py
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
$env:BSFS_CHECKPOINT_DIR="C:\gutelligence\cv_model\Model-Dev-main\checkpoints_clean_split_highres_380_tuned300"
$env:BSFS_IMG_SIZE="380"
$env:BSFS_RESIZE_SIZE="420"
.\.venv\Scripts\python.exe fine_tune_highres.py --checkpoint checkpoints_clean_split_tuned_300\bsfs_efficientnet_b4_final.pth
.\.venv\Scripts\python.exe evaluate.py --checkpoint checkpoints_clean_split_highres_380_tuned300\bsfs_efficientnet_b4_final.pth
.\.venv\Scripts\python.exe analyze_errors.py --checkpoint checkpoints_clean_split_highres_380_tuned300\bsfs_efficientnet_b4_final.pth --out error_analysis_clean_split_highres_380_tuned300 --max-images-per-bucket 30
$env:BSFS_CHECKPOINT_DIR="C:\gutelligence\cv_model\Model-Dev-main\checkpoints_clean_split_convnext_tiny"
.\.venv\Scripts\python.exe train_convnext_tiny.py
.\.venv\Scripts\python.exe evaluate_mixed_ensemble.py --checkpoint-a checkpoints_clean_split_convnext_tiny\bsfs_convnext_tiny_final.pth --type-a convnext_tiny --checkpoint-b checkpoints_clean_split_tuned_300\bsfs_efficientnet_b4_final.pth --type-b efficientnet_b4 --name convnext_tiny_tuned300_valid_weighted
.\.venv\Scripts\python.exe evaluate_mixed_ensemble.py --checkpoint-a checkpoints_clean_split_convnext_tiny\bsfs_convnext_tiny_final.pth --type-a convnext_tiny --checkpoint-b checkpoints_clean_split_tuned_300_macro_f1\bsfs_efficientnet_b4_final.pth --type-b efficientnet_b4 --name convnext_tiny_macro_f1_valid_weighted
.\.venv\Scripts\python.exe optimize_single_postprocess.py --checkpoint checkpoints_clean_split_convnext_tiny\bsfs_convnext_tiny_final.pth --model-type convnext_tiny --name convnext_tiny_valid_postprocess
.\.venv\Scripts\python.exe bsfs_inference.py --image "GuTelligence-StoMy-Clean-Split\test\Type 1\test__type1_1_jpg.rf.a203c074ce6e75b9266fe03af3169bbe.jpg" --mode best_accuracy
```

### Product Metric Update: Type 3/4 Relaxed Correctness

Date: 2026-06-02

Files changed:

- `evaluate.py`
- `analyze_errors.py`
- `bsfs_inference.py`
- `evaluate_relaxed_metrics.py`
- `model_registry.json`
- `INFERENCE.md`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`
- `checkpoints_clean_split_convnext_tiny/relaxed_metrics_type34.md`
- `checkpoints_clean_split_convnext_tiny/relaxed_metrics_type34.json`

Decision:

- Product interpretation now treats Type 3 and Type 4 as mutually acceptable normal-range states.
- The raw 7-class prediction and confidence are still preserved.
- Inference now reports both:
  - `confidence`: raw top-1 probability for the strict model prediction.
  - `clinical_confidence`: product-facing probability; for Type 3/4 this is `P(Type 3) + P(Type 4)`.

Current ConvNeXt-Tiny metrics under this rule:

```text
Strict 7-class accuracy: 48.15% (104/216)
Type 3/4 relaxed accuracy: 57.41% (124/216)
Gain from relaxed Type 3/4 rule: +9.26 percentage points
Type 3/4 swap errors recovered: 20
Mean confidence, all predictions: 0.7394
Mean confidence, recovered Type 3/4 swaps: 0.8271
```

Example inference behavior:

- Raw prediction can remain `Type 4`.
- Product-facing `clinical_pred_label` becomes `Type 3/4 normal-range`.
- `confidence` still shows the raw top-1 probability.
- `clinical_confidence` shows the combined Type 3/4 probability.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile evaluate.py analyze_errors.py bsfs_inference.py evaluate_relaxed_metrics.py
.\.venv\Scripts\python.exe evaluate_relaxed_metrics.py --predictions checkpoints_clean_split_convnext_tiny\all_predictions_tta.csv --out checkpoints_clean_split_convnext_tiny\relaxed_metrics_type34.md
.\.venv\Scripts\python.exe bsfs_inference.py --image "GuTelligence-StoMy-Clean-Split\test\Type 3\test__6xjvw61qgtdb1_jpg.rf.da720043a496c69c4c03cadcf12008df.jpg" --mode best_accuracy
```

### Phase 3 Product-Label Experiment: Type 3/4 Merged 6-Class ConvNeXt

Date: 2026-06-02

Files changed:

- `train_convnext_tiny_type34_merged.py`
- `checkpoints_clean_split_convnext_tiny_type34_merged/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Context:

- CTO and model development agreed that Type 3 and Type 4 are both normal-range states and should be mutually acceptable in the product interpretation layer.
- This experiment tests whether training directly on a 6-class product label space improves over the 7-class ConvNeXt model plus Type 3/4 relaxed evaluation.
- Original image folders were not changed; labels were remapped in the Dataset wrapper.

6-class label mapping:

```text
Type 1 -> Type 1
Type 2 -> Type 2
Type 3 -> Type 3/4 normal-range
Type 4 -> Type 3/4 normal-range
Type 5 -> Type 5
Type 6 -> Type 6
Type 7 -> Type 7
```

Training setup:

- Backbone: ImageNet-pretrained ConvNeXt-Tiny.
- Input: 300x300, resize 340.
- Head warmup: 6 epochs.
- Full fine-tune: requested 16 epochs, early-stopped at epoch 10.
- Best full checkpoint: epoch 4.
- Output dir: `checkpoints_clean_split_convnext_tiny_type34_merged/`.

Results:

```text
6-class single-view test accuracy: 54.63%
6-class single-view macro F1: 0.4927
6-class TTA test accuracy: 52.78%
6-class TTA macro F1: 0.4744
6-class TTA adjacent accuracy: 84.26%
6-class TTA MAE: 0.6944
```

Comparison to current 7-class ConvNeXt plus product relaxed rule:

```text
7-class ConvNeXt + Type 3/4 relaxed accuracy: 57.41%
6-class merged ConvNeXt single-view accuracy: 54.63%
6-class merged ConvNeXt TTA accuracy: 52.78%
```

Decision:

- Do not replace the current `best_accuracy` registry mode yet.
- The 7-class ConvNeXt model with Type 3/4 relaxed product interpretation remains the best current product-facing candidate.
- The 6-class training direction is still plausible, but this first run underperformed; TTA hurt the 6-class model, and the model still confuses the normal-range class with Type 5/6.

Next options:

- Evaluate a no-TTA/single-center-crop inference mode for the 6-class product model.
- Try a shorter full fine-tune or stronger regularization because train accuracy rose quickly while validation plateaued.
- Consider using the trained 7-class ConvNeXt as initialization for a 6-class head only if we add a safe checkpoint conversion path.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile train_convnext_tiny_type34_merged.py
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
$env:BSFS_CHECKPOINT_DIR="C:\gutelligence\cv_model\Model-Dev-main\checkpoints_clean_split_convnext_tiny_type34_merged"
.\.venv\Scripts\python.exe train_convnext_tiny_type34_merged.py
```

### Product Grouping Experiment: 4-Class BSFS Status

Date: 2026-06-02

Files changed:

- `evaluate_grouped_metrics.py`
- `train_convnext_tiny_4class_grouped.py`
- `bsfs_inference.py`
- `model_registry.json`
- `INFERENCE.md`
- `checkpoints_clean_split_convnext_tiny/grouped_metrics_4class.md`
- `checkpoints_clean_split_convnext_tiny/grouped_metrics_4class.json`
- `checkpoints_clean_split_convnext_tiny_4class_grouped/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Product grouping:

```text
Type 1 + Type 2 -> Type 1/2 hard
Type 3 + Type 4 -> Type 3/4 normal-range
Type 5 + Type 6 -> Type 5/6 soft-loose
Type 7          -> Type 7 watery
```

Post-hoc grouped metrics from the current 7-class ConvNeXt model:

```text
Source predictions: checkpoints_clean_split_convnext_tiny/all_predictions_tta.csv
Grouped accuracy: 70.37% (152/216)
Grouped macro F1: 0.6292
Mean grouped confidence: 0.8370
Mean grouped confidence for correct predictions: 0.8496
```

Direct 4-class ConvNeXt training:

```text
Output dir: checkpoints_clean_split_convnext_tiny_4class_grouped/
Center-crop 4-class accuracy: 64.81%
Center-crop macro F1: 0.5934
TTA 4-class accuracy: 64.81%
TTA macro F1: 0.5914
TTA adjacent accuracy: 91.67%
TTA MAE: 0.4398
```

Decision:

- Do not replace the current 7-class ConvNeXt model with the direct 4-class model.
- Current best product-facing strategy is:
  - keep the 7-class ConvNeXt model,
  - expose raw 7-class prediction and raw confidence,
  - expose Type 3/4 relaxed clinical interpretation,
  - expose the 4-class grouped product status and grouped confidence.
- `bsfs_inference.py` now returns:
  - `group_pred_label`
  - `group_confidence`
  - `group_interpretation.group_probabilities`

Remaining issue:

- Type 7 watery remains weak even under grouping. In the post-hoc grouped confusion matrix, true Type 7 is often predicted as `Type 5/6 soft-loose`.
- This is product-relevant because Type 7 may be a higher-risk state than Type 5/6.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile evaluate_grouped_metrics.py train_convnext_tiny_4class_grouped.py bsfs_inference.py
.\.venv\Scripts\python.exe evaluate_grouped_metrics.py --predictions checkpoints_clean_split_convnext_tiny\all_predictions_tta.csv --out checkpoints_clean_split_convnext_tiny\grouped_metrics_4class.md
$env:BSFS_DATA_ROOT="C:\gutelligence\cv_model\Model-Dev-main\GuTelligence-StoMy-Clean-Split"
$env:BSFS_CHECKPOINT_DIR="C:\gutelligence\cv_model\Model-Dev-main\checkpoints_clean_split_convnext_tiny_4class_grouped"
.\.venv\Scripts\python.exe train_convnext_tiny_4class_grouped.py
$img = (Get-ChildItem "GuTelligence-StoMy-Clean-Split\test\Type 6" -File | Select-Object -First 1).FullName
.\.venv\Scripts\python.exe bsfs_inference.py --image $img --mode best_accuracy
```

### 4-Class Grouped Error Analysis, Selective Prediction, and Type 7 Experiments

Date: 2026-06-02

Files changed:

- `analyze_grouped_errors.py`
- `evaluate_grouped_selective.py`
- `evaluate_type7_gate.py`
- `train_type7_binary.py`
- `evaluate_type7_cascade.py`
- `bsfs_inference.py`
- `model_registry.json`
- `INFERENCE.md`
- `error_analysis_grouped_4class/`
- `selective_results_grouped/`
- `type7_gate_results/`
- `checkpoints_type7_binary/`
- `type7_cascade_results/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Grouped error analysis:

```text
Prediction CSV: checkpoints_clean_split_convnext_tiny/all_predictions_tta.csv
Grouped accuracy: 70.37% (152/216)
Grouped errors: 64
High-confidence grouped errors (group confidence >= 0.70): 48
Main high-risk issue: true Type 7 watery is often predicted as Type 5/6 soft-loose.
```

Grouped selective prediction:

```text
Selected thresholds on valid: confidence >= 0.25, margin >= 0.12
Valid coverage: 87.85%
Valid accuracy: 79.26%
Test coverage: 94.44%
Test accuracy on kept: 69.61%
Test macro F1 on kept: 0.6177
Test Type 7 recall on kept: 17.65%
```

Decision:

- Do not use grouped selective thresholds as a 90% path.
- Validation threshold gains did not generalize to test.
- Selective prediction reduced test accuracy from the full-coverage grouped baseline `70.37%` to `69.61%` on kept samples.

Type 7 probability gate:

```text
Best valid accuracy: 75.23%
Best valid Type 7 recall: 94.74%
Test accuracy: 66.67%
Test Type 7 recall: 36.84%
Test Type 7 precision: 31.82%
```

Decision:

- Do not use the Type 7 probability gate.
- It overfit validation behavior and lowered test accuracy.

Type 5/6 vs Type 7 binary specialist:

```text
Output dir: checkpoints_type7_binary/
Initialized from: checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth
TTA binary accuracy: 82.86%
TTA macro F1: 0.6842
TTA Type 7 recall: 42.11%
TTA Type 7 precision: 53.33%
```

Type 7 cascade evaluation:

```text
Base model: checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth
Binary specialist: checkpoints_type7_binary/bsfs_type7_binary_final.pth
Selected valid cascade: raw_top1_mapping + binary Type 7 threshold 0.12
Valid grouped accuracy: 77.10%
Valid Type 7 recall: 100.00%
Test grouped accuracy: 65.28%
Test macro F1: 0.6099
Test Type 7 recall: 36.84%
Test Type 7 precision: 23.33%
```

Decision:

- Do not use the Type 7 cascade as the primary product model.
- The specialist increases Type 7 recall but introduces too many false Type 7 predictions and lowers overall grouped accuracy.
- Current best product-facing model remains the 7-class ConvNeXt-Tiny with raw top-1 label mapped to the 4 product groups.

Inference wrapper update:

- `bsfs_inference.py` now returns `group_pred_label` using raw 7-class top-1 label mapped to 4 product groups, matching the measured `70.37%` grouped metric.
- It still exposes summed group probabilities via `group_interpretation.group_probabilities`.
- `model_registry.json` was updated to version `2026-06-02-grouped-product-v2`.
- Registry records raw-top1-mapped grouped accuracy `70.37%` and group-sum-argmax grouped accuracy `68.98%`.

Conclusion for the 90% target:

- The current clean split does not support a 90% product accuracy path through thresholding, selective prediction, or a simple Type 7 specialist.
- The next meaningful accuracy work should move to ROI/segmentation, device-native data, more Type 7 examples, and event-level multi-frame aggregation.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile evaluate_grouped_selective.py evaluate_type7_gate.py train_type7_binary.py evaluate_type7_cascade.py bsfs_inference.py
.\.venv\Scripts\python.exe evaluate_type7_gate.py --checkpoint checkpoints_clean_split_convnext_tiny\bsfs_convnext_tiny_final.pth --model-type convnext_tiny --name convnext_tiny_type7_gate
.\.venv\Scripts\python.exe train_type7_binary.py --init-checkpoint checkpoints_clean_split_convnext_tiny\bsfs_convnext_tiny_final.pth
.\.venv\Scripts\python.exe evaluate_type7_cascade.py --base-checkpoint checkpoints_clean_split_convnext_tiny\bsfs_convnext_tiny_final.pth --base-model-type convnext_tiny --binary-checkpoint checkpoints_type7_binary\bsfs_type7_binary_final.pth --name convnext_tiny_type7_binary_cascade
```

### No-New-Data Accuracy Improvement Attempts

Date: 2026-06-02

Files changed:

- `evaluate_roi_preprocessing.py`
- `evaluate_grouped_mixed_ensemble.py`
- `train_convnext_tiny.py`
- `roi_preprocessing_results/convnext_tiny_roi_eval/`
- `checkpoints_clean_split_convnext_tiny_reg_seed2/`
- `checkpoints_clean_split_convnext_tiny_seed3/`
- `ensemble_results/convnext_tiny_best_plus_reg_seed2/`
- `ensemble_results/convnext_tiny_best_plus_seed3/`
- `grouped_ensemble_results/convnext_tiny_best_plus_seed3_grouped/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

ROI preprocessing diagnostic:

```text
Script: evaluate_roi_preprocessing.py
Best valid preprocessing: standard_center
Valid standard_center grouped accuracy: 77.10%
Valid best ROI grouped accuracy: 76.64%
Test standard_center grouped accuracy: 70.37%
Test standard_5view_tta grouped accuracy: 70.37%
```

Decision:

- Do not train a ROI-preprocessed classifier from the current automatic ROI heuristic.
- Simple threshold/texture ROI crops did not beat standard preprocessing on valid.

Regularized ConvNeXt seed2:

```text
Output dir: checkpoints_clean_split_convnext_tiny_reg_seed2/
Seed: 20260602
Changes: shorter schedule, stronger weight decay, lower LR, higher label smoothing
TTA strict accuracy: 40.74%
TTA macro F1: 0.4136
TTA adjacent accuracy: 79.17%
```

Decision:

- Reject this candidate. Stronger regularization underfit or selected a poorer solution.
- Mixed ensemble with the current best ConvNeXt also failed: test strict accuracy `42.13%`.

ConvNeXt seed3 with baseline hyperparameters:

```text
Output dir: checkpoints_clean_split_convnext_tiny_seed3/
Seed: 20260603
TTA strict accuracy: 47.69%
TTA macro F1: 0.4817
Type 3/4 relaxed accuracy: 56.48%
4-class grouped accuracy: 68.52%
4-class grouped macro F1: 0.6289
```

Decision:

- Seed3 is close to the current best model on strict accuracy and macro F1, but it does not improve product grouped accuracy.
- Keep the current best ConvNeXt checkpoint as primary.

Grouped ensemble of current best ConvNeXt + seed3:

```text
Script: evaluate_grouped_mixed_ensemble.py
Selected on valid grouped product score
Selected weights: current best 0.25, seed3 0.75
Valid grouped accuracy: 79.44%
Test grouped accuracy: 67.59%
Test grouped macro F1: 0.6185
```

Decision:

- Reject the grouped ensemble. It overfits the validation split and underperforms the current `70.37%` grouped baseline on test.
- Do not continue additional valid-selected weighting or threshold searches on the current split without a stronger validation protocol.

Conclusion:

- Within the current dataset and split, the best product-facing model remains:
  - `checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth`
  - raw 7-class top-1 mapped to 4 product groups
  - 4-class grouped accuracy `70.37%`
- The highest-signal next no-new-data direction is not more valid-tuned ensembling. It is to improve validation protocol, e.g. repeated clean splits or cross-validation, before trusting small model-only gains.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile evaluate_roi_preprocessing.py evaluate_grouped_mixed_ensemble.py train_convnext_tiny.py
.\.venv\Scripts\python.exe evaluate_roi_preprocessing.py --checkpoint checkpoints_clean_split_convnext_tiny\bsfs_convnext_tiny_final.pth --model-type convnext_tiny --name convnext_tiny_roi_eval
$env:BSFS_CHECKPOINT_DIR="C:\gutelligence\cv_model\Model-Dev-main\checkpoints_clean_split_convnext_tiny_seed3"
$env:BSFS_SEED="20260603"
.\.venv\Scripts\python.exe train_convnext_tiny.py
.\.venv\Scripts\python.exe evaluate_grouped_mixed_ensemble.py --checkpoint-a checkpoints_clean_split_convnext_tiny\bsfs_convnext_tiny_final.pth --type-a convnext_tiny --checkpoint-b checkpoints_clean_split_convnext_tiny_seed3\bsfs_convnext_tiny_final.pth --type-b convnext_tiny --name convnext_tiny_best_plus_seed3_grouped
```

### Repeated Clean Split Validation Protocol

Date: 2026-06-02

Files changed:

- `create_repeated_clean_splits.py`
- `materialize_clean_split_from_manifest.py`
- `summarize_repeated_splits.py`
- `repeated_clean_splits/`
- `GuTelligence-StoMy-Clean-Split-Seed20260602/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Why this was added:

- Several recent experiments improved validation metrics but failed on test.
- The current clean split is leakage-safe but still only one split.
- Repeated leakage-safe splits let us estimate whether future improvements are stable or just artifacts of one validation/test partition.

Repeated split generation:

```text
Input manifest: dataset_audit/manifest.csv
Total records: 2149
Connected components: 299
Seeds: 20260601, 20260602, 20260603, 20260604, 20260605
Ratios: train 80%, valid 10%, test 10%
Leakage rule: source_id and exact SHA-256 components stay within one split
```

Leakage result:

```text
seed 20260601: source leaks 0, sha leaks 0
seed 20260602: source leaks 0, sha leaks 0
seed 20260603: source leaks 0, sha leaks 0
seed 20260604: source leaks 0, sha leaks 0
seed 20260605: source leaks 0, sha leaks 0
```

Assignment diversity:

```text
Pairwise assignment difference range: 12.94% - 17.96%
Total images per manifest: 2149
Split sizes per seed: train 1719, valid 214, test 216
Class counts are identical across seeds.
```

Materialization dry-run:

```text
Manifest: repeated_clean_splits/seed_20260602/split_manifest.csv
Output: GuTelligence-StoMy-Clean-Split-Seed20260602/
Copied images: 2149
Report: GuTelligence-StoMy-Clean-Split-Seed20260602/_split_report/summary.md
```

Decision:

- Use repeated clean split manifests as the next validation protocol before trusting new model-only gains.
- Do not replace the current primary model yet.
- Next model-development step should train/evaluate a lightweight candidate across at least 3 repeated splits and report mean/std for strict, Type 3/4 relaxed, grouped 4-class, macro F1, adjacent accuracy, and Type 7 recall.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile create_repeated_clean_splits.py materialize_clean_split_from_manifest.py summarize_repeated_splits.py
.\.venv\Scripts\python.exe create_repeated_clean_splits.py --seeds 20260601 20260602 20260603 20260604 20260605
.\.venv\Scripts\python.exe summarize_repeated_splits.py --split-dir repeated_clean_splits
.\.venv\Scripts\python.exe materialize_clean_split_from_manifest.py --manifest repeated_clean_splits\seed_20260602\split_manifest.csv --out GuTelligence-StoMy-Clean-Split-Seed20260602 --overwrite
```

### Repeated Split Lightweight ConvNeXt Experiment

Date: 2026-06-02

Files changed:

- `run_repeated_split_experiment.py`
- `repeated_split_experiments/smoke_convnext_tiny_repeated/`
- `repeated_split_experiments/convnext_tiny_light_3split/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Purpose:

- Validate the new repeated split protocol with real training runs.
- Establish a mean/std baseline for a lightweight ConvNeXt schedule.
- Avoid trusting a single valid/test split after repeated valid gains failed to generalize.

Smoke test:

```text
Experiment: smoke_convnext_tiny_repeated
Seed: 20260601
Schedule: head 1 epoch, full 1 epoch
Result: flow completed successfully
Temporary materialized split removed
Large .pth checkpoint files removed
```

Main repeated split experiment:

```text
Experiment: convnext_tiny_light_3split
Seeds: 20260601, 20260602, 20260603
Schedule: head 4 epochs, full 8 epochs, patience 4
Temporary materialized split directories: removed after each seed
Large .pth checkpoint files: removed after metrics were written
```

Mean/std results across 3 repeated splits:

```text
Strict accuracy           mean 0.4645 | std 0.0777 | min 0.3750 | max 0.5139
Macro F1                  mean 0.4188 | std 0.0823 | min 0.3262 | max 0.4833
Type 3/4 relaxed accuracy mean 0.5139 | std 0.0516 | min 0.4583 | max 0.5602
4-class grouped accuracy  mean 0.6420 | std 0.0586 | min 0.5787 | max 0.6944
4-class grouped macro F1  mean 0.5813 | std 0.1010 | min 0.4803 | max 0.6823
Adjacent accuracy         mean 0.7917 | std 0.0379 | min 0.7593 | max 0.8333
MAE                       mean 0.9954 | std 0.1895 | min 0.8333 | max 1.2037
Type 7 recall             mean 0.5263 | std 0.3646 | min 0.3158 | max 0.9474
Type 7 precision          mean 0.4203 | std 0.1705 | min 0.2609 | max 0.6000
```

Interpretation:

- The lightweight schedule does not beat the current primary ConvNeXt model.
- Metric variance across leakage-safe repeated splits is large.
- Type 7 recall is especially unstable, ranging from `31.58%` to `94.74%`.
- Future model changes should be accepted only if they improve repeated-split mean metrics and keep variance acceptable.

Current decision:

- Do not replace the current primary model.
- Keep `checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth` as the best product-facing checkpoint.
- Use `repeated_split_experiments/convnext_tiny_light_3split/summary.md` as the first repeated-split baseline report.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile run_repeated_split_experiment.py materialize_clean_split_from_manifest.py train_convnext_tiny.py
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name smoke_convnext_tiny_repeated --seeds 20260601 --head-epochs 1 --full-epochs 1 --patience 1
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name convnext_tiny_light_3split --seeds 20260601 20260602 20260603 --head-epochs 4 --full-epochs 8 --patience 4
```

### Full ConvNeXt-Tiny Repeated Split Baseline

Date: 2026-06-03

Files changed:

- `run_repeated_split_experiment.py`
- `repeated_split_experiments/convnext_tiny_full_3split/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Correction:

- An initial run used the experiment name `convnext_tiny_full_3split` but inherited the runner's lightweight defaults.
- That mislabeled result directory was deleted.
- `run_repeated_split_experiment.py` defaults were updated to match the full ConvNeXt-Tiny schedule: head `8`, full `22`, patience `8`.
- The full baseline was rerun with explicit parameters.

Experiment:

```text
Name: convnext_tiny_full_3split
Seeds: 20260601, 20260602, 20260603
Schedule: head 8 epochs, full 22 epochs, patience 8
Temporary materialized split directories: removed after each seed
Large .pth checkpoint files: removed after metrics were written
```

Mean/std results across 3 repeated splits:

```text
Strict accuracy           mean 0.4244 | std 0.0936 | min 0.3241 | max 0.5093
Macro F1                  mean 0.3862 | std 0.1087 | min 0.2666 | max 0.4790
Type 3/4 relaxed accuracy mean 0.4846 | std 0.0696 | min 0.4120 | max 0.5509
4-class grouped accuracy  mean 0.6219 | std 0.0813 | min 0.5370 | max 0.6991
4-class grouped macro F1  mean 0.5607 | std 0.1245 | min 0.4401 | max 0.6888
Adjacent accuracy         mean 0.7948 | std 0.0579 | min 0.7361 | max 0.8519
MAE                       mean 1.0262 | std 0.2176 | min 0.8102 | max 1.2454
Type 7 recall             mean 0.5263 | std 0.3646 | min 0.3158 | max 0.9474
Type 7 precision          mean 0.3599 | std 0.2105 | min 0.2069 | max 0.6000
```

Comparison to lightweight repeated baseline:

```text
Light schedule grouped accuracy mean: 0.6420
Full schedule grouped accuracy mean : 0.6219
Light schedule strict accuracy mean : 0.4645
Full schedule strict accuracy mean  : 0.4244
```

Interpretation:

- Full schedule did not improve repeated-split mean performance.
- The longer schedule appears more sensitive to split instability and/or overfitting.
- The current primary model still remains the best product-facing checkpoint on the original clean split, but repeated-split evaluation shows that single-split metrics are optimistic and unstable.

Decision:

- Do not replace the current primary model.
- Do not prioritize longer ConvNeXt-Tiny training as the next accuracy path.
- Next model-improvement direction should be a loss/objective change, such as ordinal-aware loss or 4-class + 7-class multi-task learning, evaluated with repeated splits.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile run_repeated_split_experiment.py train_convnext_tiny.py materialize_clean_split_from_manifest.py
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name convnext_tiny_full_3split --seeds 20260601 20260602 20260603 --head-epochs 8 --full-epochs 22 --patience 8
```

### Ordinal-Aware ConvNeXt Repeated Split Experiment

Date: 2026-06-03

Files changed:

- `train_convnext_tiny_ordinal.py`
- `run_repeated_split_experiment.py`
- `repeated_split_experiments/smoke_convnext_tiny_ordinal/`
- `repeated_split_experiments/convnext_tiny_ordinal_light_3split/`
- `DEVELOPMENT_LOG.md`
- `MODEL_DEVELOPMENT_PLAN.md`

Model change:

```text
Base architecture: ConvNeXt-Tiny, 7-class output
Loss: CrossEntropy(label_smoothing=0.02) + ordinal_lambda * expected normalized class distance
Default ordinal_lambda: 0.35
Training schedule for comparison: head 4 epochs, full 8 epochs, patience 4
Seeds: 20260601, 20260602, 20260603
```

Repeated split results:

```text
Strict accuracy           mean 0.4707 | std 0.0560 | min 0.4074 | max 0.5139
Macro F1                  mean 0.4213 | std 0.0649 | min 0.3530 | max 0.4823
Type 3/4 relaxed accuracy mean 0.5262 | std 0.0325 | min 0.4954 | max 0.5602
4-class grouped accuracy  mean 0.6435 | std 0.0561 | min 0.5833 | max 0.6944
4-class grouped macro F1  mean 0.5845 | std 0.0947 | min 0.4914 | max 0.6807
Adjacent accuracy         mean 0.7840 | std 0.0533 | min 0.7315 | max 0.8380
MAE                       mean 1.0046 | std 0.2092 | min 0.8194 | max 1.2315
Type 7 recall             mean 0.5263 | std 0.3646 | min 0.3158 | max 0.9474
Type 7 precision          mean 0.4380 | std 0.1280 | min 0.3333 | max 0.5806
```

Comparison to non-ordinal lightweight baseline:

```text
Strict accuracy           +0.0062
Macro F1                  +0.0025
Type 3/4 relaxed accuracy +0.0123
4-class grouped accuracy  +0.0015
4-class grouped macro F1  +0.0032
Adjacent accuracy         -0.0077
MAE                       +0.0093
Type 7 recall             +0.0000
Type 7 precision          +0.0177
```

Interpretation:

- Ordinal-aware loss gives a small positive mean improvement in strict, relaxed, grouped accuracy, grouped macro F1, and Type 7 precision.
- The improvement is too small to justify replacing the current primary model.
- Adjacent accuracy and MAE slightly worsened, so the current ordinal penalty formulation is not yet clearly better.
- This direction remains useful but needs refinement, such as lambda sweep or multi-task grouping, before registry consideration.

Decision:

- Do not replace the current primary model.
- Keep `train_convnext_tiny_ordinal.py` as an experimental training objective.
- Next objective-change experiment should be 4-class product head + 7-class auxiliary head, evaluated with the same repeated split runner.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile train_convnext_tiny_ordinal.py run_repeated_split_experiment.py
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name smoke_convnext_tiny_ordinal --train-script train_convnext_tiny_ordinal.py --seeds 20260601 --head-epochs 1 --full-epochs 1 --patience 1
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name convnext_tiny_ordinal_light_3split --train-script train_convnext_tiny_ordinal.py --seeds 20260601 20260602 20260603 --head-epochs 4 --full-epochs 8 --patience 4
```

### Repository Cleanup

Date: 2026-06-02

Reason:

- The working directory had accumulated many rejected experiment checkpoints, figure directories, error-analysis copies, and dry-run materialized datasets.
- These artifacts made the project difficult to navigate and consumed several GB.

Deleted directories:

```text
checkpoints/
checkpoints_clean_split/
checkpoints_clean_split_tuned_300_composite/
checkpoints_clean_split_highres_380_tuned300/
checkpoints_clean_split_ordinal_soft_tuned300/
checkpoints_clean_split_type3_ft_tuned300/
checkpoints_neighbor_234_tuned300_init/
checkpoints_neighbor_234_imagenet/
checkpoints_clean_split_convnext_tiny_4class_grouped/
checkpoints_clean_split_convnext_tiny_reg_seed2/
checkpoints_clean_split_convnext_tiny_seed3/
checkpoints_clean_split_convnext_tiny_type34_merged/
checkpoints_type7_binary/
GuTelligence-StoMy-Clean-Split-Seed20260602/
ensemble_results/
grouped_ensemble_results/
hierarchical_results/
postprocess_results/
roi_preprocessing_results/
type7_gate_results/
type7_cascade_results/
selective_results_grouped/
training_runs/
inference_samples/
figures*/
error_analysis_clean_split*/
__pycache__/
```

Approximate space freed:

```text
3586.6 MB
```

Kept directories and artifacts:

```text
GuTelligence-StoMy-First-Project-2/
GuTelligence-StoMy-Clean-Split/
dataset_audit/
repeated_clean_splits/
checkpoints_clean_split_convnext_tiny/
checkpoints_clean_split_tuned_300/
checkpoints_clean_split_tuned_300_macro_f1/
checkpoints_clean_split_focal_ft_tuned300/
selective_results/
error_analysis_grouped_4class/
type3_type4_review_artifacts/
```

Rationale for kept model checkpoints:

- `checkpoints_clean_split_convnext_tiny/` is the current best product-facing model.
- `checkpoints_clean_split_tuned_300/` and `checkpoints_clean_split_tuned_300_macro_f1/` are still referenced by `model_registry.json` high-coverage mode.
- `checkpoints_clean_split_focal_ft_tuned300/` is still referenced by `model_registry.json` cautious mode.

Validation after cleanup:

```text
All registry checkpoint paths exist.
bsfs_inference.py smoke test passed with registry version 2026-06-02-grouped-product-v2.
Core scripts compiled successfully.
No Python training/inference process remained running.
```

Cleanup policy going forward:

- Keep source datasets, clean split, audit reports, repeated split manifests, current registry checkpoints, and concise summary reports.
- Delete rejected checkpoints after their metrics and decisions are recorded in `DEVELOPMENT_LOG.md`.
- Prefer manifests over materialized duplicate dataset directories; materialize repeated splits only while training/evaluating.

### ConvNeXt-Tiny Multi-Task Product Group + Raw BSFS Experiment

Date: 2026-06-03

Goal:

- Test whether a product-facing 4-class group head, assisted by a 7-class raw BSFS auxiliary head, can improve grouped product accuracy without losing diagnostic raw-class quality.

Files added or updated:

```text
train_convnext_tiny_multitask.py
run_repeated_split_experiment.py
repeated_split_experiments/convnext_tiny_multitask_light_3split/
```

Model objective:

```text
Backbone: ConvNeXt-Tiny
Primary head: 4 product groups
Auxiliary head: 7 raw BSFS classes
Loss: group CE + 0.35 * raw CE
Prediction fusion: group head + 0.35 * raw head mapped into 7-class probabilities
Schedule: head 4 epochs, full 8 epochs, patience 4
Seeds: 20260601, 20260602, 20260603
```

Repeated split results:

```text
Strict accuracy           mean 0.4090 | std 0.0657 | min 0.3380 | max 0.4676
Macro F1                  mean 0.3571 | std 0.0539 | min 0.3090 | max 0.4153
Type 3/4 relaxed accuracy mean 0.4985 | std 0.0630 | min 0.4491 | max 0.5694
4-class grouped accuracy  mean 0.6466 | std 0.0457 | min 0.6157 | max 0.6991
4-class grouped macro F1  mean 0.5789 | std 0.0842 | min 0.5196 | max 0.6753
Adjacent accuracy         mean 0.7701 | std 0.0486 | min 0.7222 | max 0.8194
MAE                       mean 1.1481 | std 0.1250 | min 1.0231 | max 1.2731
Type 7 recall             mean 0.4912 | std 0.3950 | min 0.2632 | max 0.9474
Type 7 precision          mean 0.4305 | std 0.1438 | min 0.2941 | max 0.5806
```

Comparison to lightweight ConvNeXt repeated baseline:

```text
4-class grouped accuracy  +0.0046
Strict accuracy           -0.0556
Macro F1                  -0.0617
Type 7 recall             -0.0351
Type 7 precision          +0.0102
```

Interpretation:

- Multi-task learning gave a very small grouped product accuracy gain and slightly lower grouped accuracy variance.
- The raw 7-class diagnostic metrics degraded materially, especially strict accuracy and macro F1.
- This objective is not suitable for replacing the current primary model in its current weighting.

Decision:

- Do not update `model_registry.json`.
- Keep `checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth` as the current primary model.
- Continue only as an experimental branch by increasing raw-loss and raw-fusion weights to reduce raw-class degradation while checking whether grouped accuracy remains stable.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile train_convnext_tiny_multitask.py run_repeated_split_experiment.py bsfs_inference.py
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name smoke_convnext_tiny_multitask --train-script train_convnext_tiny_multitask.py --seeds 20260601 --head-epochs 1 --full-epochs 1 --patience 1
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name convnext_tiny_multitask_light_3split --train-script train_convnext_tiny_multitask.py --seeds 20260601 20260602 20260603 --head-epochs 4 --full-epochs 8 --patience 4
```

### Repeated Split Runner Path-Length Fix

Date: 2026-06-05

Reason:

- A long experiment name combined with long Roboflow-style image filenames caused Windows file-copy failure while materializing repeated split data.
- The failed run did not start training; it failed during `materialize_clean_split_from_manifest.py`.

Files updated:

```text
run_repeated_split_experiment.py
materialize_clean_split_from_manifest.py
```

Changes:

- Replaced long temporary split directory names with short deterministic hash-based names, such as `_tmp_rs_<hash>_<seed>`.
- Added a defensive `dest.parent.mkdir(parents=True, exist_ok=True)` before copying each materialized image.
- Removed the failed partial experiment directory and stale temporary split directory before rerunning.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile materialize_clean_split_from_manifest.py run_repeated_split_experiment.py train_convnext_tiny_multitask.py
```

### ConvNeXt-Tiny Multi-Task Raw-Weighted Experiment

Date: 2026-06-05

Goal:

- Test whether stronger raw BSFS supervision can recover the raw 7-class degradation observed in the initial multi-task model while preserving grouped product accuracy.

Experiment:

```text
Experiment directory: repeated_split_experiments/convnext_tiny_multitask_raw065_fuse055_light_3split/
Train script: train_convnext_tiny_multitask.py
Raw loss weight: 0.65
Raw fusion weight: 0.55
Schedule: head 4 epochs, full 8 epochs, patience 4
Seeds: 20260601, 20260602, 20260603
```

Repeated split results:

```text
Strict accuracy           mean 0.3781 | std 0.0932 | min 0.2917 | max 0.4769
Macro F1                  mean 0.3194 | std 0.0867 | min 0.2677 | max 0.4194
Type 3/4 relaxed accuracy mean 0.4676 | std 0.0893 | min 0.4028 | max 0.5694
4-class grouped accuracy  mean 0.6219 | std 0.0473 | min 0.5880 | max 0.6759
4-class grouped macro F1  mean 0.5345 | std 0.0846 | min 0.4724 | max 0.6309
Adjacent accuracy         mean 0.7577 | std 0.0374 | min 0.7361 | max 0.8009
MAE                       mean 1.1852 | std 0.1530 | min 1.0093 | max 1.2870
Type 7 recall             mean 0.4737 | std 0.4178 | min 0.1579 | max 0.9474
Type 7 precision          mean 0.4505 | std 0.1996 | min 0.2308 | max 0.6207
```

Comparison:

```text
Versus initial multi-task raw035/fuse035:
  grouped accuracy 0.6219 vs 0.6466
  strict accuracy  0.3781 vs 0.4090
  macro F1         0.3194 vs 0.3571
  Type 7 recall    0.4737 vs 0.4912

Versus lightweight ConvNeXt repeated baseline:
  grouped accuracy 0.6219 vs 0.6420
  strict accuracy  0.3781 vs 0.4645
  macro F1         0.3194 vs 0.4188
```

Decision:

- Reject this raw-weighted multi-task configuration.
- Do not update `model_registry.json`.
- Increasing both raw loss and raw fusion did not recover diagnostic quality and reduced grouped product accuracy.
- If multi-task is continued, the next test should decouple prediction from group fusion, such as raw-head-only inference with an auxiliary group loss.

Cleanup and process state:

```text
No temporary split directories remained.
No .pth files remained in the rejected repeated experiment.
No Python training process remained running.
```

## 2026-06-23 - Model architecture visualization

Goal:

- Add model-structure visibility for engineering and CTO review.
- Keep Hugging Face MVP simple and reliable.
- Provide a separate Netron path for graph-level inspection.

Files updated:

```text
hf_space_mvp/app.py
hf_space_mvp/requirements.txt
requirements.txt
README.md
HUGGINGFACE_SPACE_DEPLOYMENT.md
MODEL_DEVELOPMENT_PLAN.md
.gitignore
```

Files added:

```text
export_model_for_netron.py
```

Implementation:

- Added a `Model Architecture` tab to the Hugging Face Gradio app.
- Added `torchinfo` to the Space dependencies.
- The architecture tab reports:
  - device availability
  - registry version
  - ConvNeXt-Tiny backbone
  - raw 7-class BSFS output
  - 3-class product grouping
  - Type 7 continuous risk probability
  - checkpoint availability
  - total/trainable parameters
  - layer summary
- Added a local ONNX export script for Netron inspection.
- Added `model_exports/` to `.gitignore` so generated ONNX files do not enter Git.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile hf_space_mvp\app.py export_model_for_netron.py
.\.venv\Scripts\python.exe export_model_for_netron.py
```

Results:

```text
HF Space architecture summary smoke test passed.
Total parameters reported by torchinfo path: 27,825,511
ONNX export succeeded: model_exports/bsfs_convnext_tiny_product.onnx
ONNX output size: 111,386,589 bytes
```

Hugging Face Space update:

```text
app.py uploaded: https://huggingface.co/spaces/perram27/bsfs-3class-type7-risk-mvp/commit/d0154a6535eee274d4dd78ab8baac521c3d788d9
requirements.txt uploaded: https://huggingface.co/spaces/perram27/bsfs-3class-type7-risk-mvp/commit/70cf4a1674cdc7396639c00e90fff4836b4e95f0
```

### ConvNeXt-Tiny Raw-Primary Multi-Task Experiment

Date: 2026-06-05

Goal:

- Test whether multi-task learning works better when the raw 7-class BSFS head remains the only inference head, while the 4-class product group head is used only as an auxiliary training loss.
- This addresses the prior failure mode where group-head fusion damaged raw 7-class diagnostic quality.

Files updated:

```text
train_convnext_tiny_multitask.py
repeated_split_experiments/smoke_convnext_tiny_multitask_rawprimary/
repeated_split_experiments/convnext_tiny_multitask_rawprimary_group035_light_3split/
```

Code changes:

- Added `BSFS_MULTITASK_LOSS_MODE`.
  - `group_primary`: previous behavior, `group CE + raw_weight * raw CE`.
  - `raw_primary`: new behavior, `raw CE + group_weight * group CE`.
- Added `BSFS_MULTITASK_PREDICTION_MODE`.
  - `fused`: previous behavior, group/raw fused probabilities.
  - `raw_only`: new behavior, raw head probabilities only.
- Added `BSFS_GROUP_LOSS_WEIGHT`, default `0.35`.

Experiment:

```text
Loss mode: raw_primary
Prediction mode: raw_only
Group loss weight: 0.35
Schedule: head 4 epochs, full 8 epochs, patience 4
Seeds: 20260601, 20260602, 20260603
```

Repeated split results:

```text
Strict accuracy           mean 0.4028 | std 0.0745 | min 0.3241 | max 0.4722
Macro F1                  mean 0.3497 | std 0.0825 | min 0.2893 | max 0.4438
Type 3/4 relaxed accuracy mean 0.4877 | std 0.0724 | min 0.4213 | max 0.5648
4-class grouped accuracy  mean 0.6327 | std 0.0543 | min 0.5926 | max 0.6944
4-class grouped macro F1  mean 0.5763 | std 0.0955 | min 0.5017 | max 0.6840
Adjacent accuracy         mean 0.7608 | std 0.0595 | min 0.7176 | max 0.8287
MAE                       mean 1.1420 | std 0.1971 | min 0.9167 | max 1.2824
Type 7 recall             mean 0.4912 | std 0.3039 | min 0.3158 | max 0.8421
Type 7 precision          mean 0.5896 | std 0.1883 | min 0.3750 | max 0.7273
```

Comparison to lightweight ConvNeXt repeated baseline:

```text
Grouped accuracy          0.6327 vs 0.6420
Strict accuracy           0.4028 vs 0.4645
Macro F1                  0.3497 vs 0.4188
Type 3/4 relaxed accuracy 0.4877 vs 0.5139
Adjacent accuracy         0.7608 vs 0.7917
Type 7 recall             0.4912 vs 0.5263
Type 7 precision          0.5896 vs 0.4203
```

Interpretation:

- Raw-primary inference avoided the most direct group-fusion failure mode but still did not outperform the plain ConvNeXt repeated baseline.
- The only meaningful gain was Type 7 precision, but this came with lower recall and lower overall accuracy/F1.
- The auxiliary group loss is not a reliable path to the 90% product target on the current dataset.

Decision:

- Reject this raw-primary multi-task configuration for registry replacement.
- Keep the configurable script for future ablation, but deprioritize multi-task loss tuning.
- Next model-only work should either test a stronger deployment-feasible backbone, such as EfficientNetV2-S or ConvNeXt-Small, or move toward calibration/OOD/product confidence work rather than more group-loss tuning.

Verification and cleanup:

```powershell
.\.venv\Scripts\python.exe -m py_compile train_convnext_tiny_multitask.py run_repeated_split_experiment.py materialize_clean_split_from_manifest.py
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name smoke_convnext_tiny_multitask_rawprimary --train-script train_convnext_tiny_multitask.py --seeds 20260601 --head-epochs 1 --full-epochs 1 --patience 1
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name convnext_tiny_multitask_rawprimary_group035_light_3split --train-script train_convnext_tiny_multitask.py --seeds 20260601 20260602 20260603 --head-epochs 4 --full-epochs 8 --patience 4
```

```text
No temporary split directories remained.
No .pth files remained in the smoke or repeated experiment directories.
No Python training process remained running.
```

## 2026-06-09 - GitHub repository preparation

Goal:

- Prepare the cleaned product codebase for GitHub upload through terminal.
- Keep the repository focused on the current product direction: 3-class BSFS grouping with a continuous Type 7 risk probability.
- Avoid committing datasets, local IDE files, virtual environments, training logs, or large model weights.

Files updated:

```text
.gitignore
DEVELOPMENT_LOG.md
MODEL_DEVELOPMENT_PLAN.md
```

Repository policy:

- Keep source code, documentation, model registry metadata, and lightweight evaluation summaries in Git.
- Exclude local BSFS image datasets from GitHub commits.
- Exclude `.pth`, `.pt`, `.onnx`, and `.safetensors` artifacts from normal GitHub commits because model checkpoints exceed GitHub's regular file-size limit and should be distributed through Hugging Face or Git LFS.
- Exclude local `.idea`, `.vscode`, `.venv`, `__pycache__`, `.pyc`, and generated log files.

Active code intended for GitHub:

```text
bsfs_inference.py
calibrate_type7_risk.py
evaluate_3class_type7_risk.py
model.py
train_convnext_tiny_3class_type7_aux.py
model_registry.json
README.md
INFERENCE.md
requirements.txt
CTO_MODEL_PROGRESS_BRIEF.tex
```

Notes:

- The trained ConvNeXt-Tiny checkpoint remains local and is referenced by `model_registry.json`, but the binary weight file is intentionally not committed.
- The next deployment step should publish the checkpoint to a Hugging Face model repository or configure Git LFS before sharing weights through GitHub.

## 2026-06-09 - Hugging Face Space MVP package

Goal:

- Prepare an MVP Hugging Face Space deployment for the current product schema.
- Expose image upload inference through Gradio.
- Return 3-class BSFS product output and continuous Type 7 probability.

Files added:

```text
hf_space_mvp/.gitattributes
hf_space_mvp/.gitignore
hf_space_mvp/README.md
hf_space_mvp/app.py
hf_space_mvp/model_registry.json
hf_space_mvp/requirements.txt
hf_space_mvp/checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth
HUGGINGFACE_SPACE_DEPLOYMENT.md
```

Implementation:

- Built a standalone Gradio app in `hf_space_mvp/app.py`.
- Reused the current ConvNeXt-Tiny 7-class checkpoint and mapped raw output to the product 3-class schema.
- Preserved Type 7 probability as a continuous risk signal with no fixed alert threshold.
- Configured Git LFS tracking for `.pth`, `.pt`, `.onnx`, and `.safetensors` files inside the Space repo.
- Prepared the Space directory so it can be initialized and pushed as a standalone Hugging Face Space repo after authentication.
- Kept the checkpoint out of the normal GitHub code repository; it is intended for Hugging Face Git LFS.

Local verification:

```powershell
cd Model-Dev-main\hf_space_mvp
$env:PYTHONPATH='.'
@'
from pathlib import Path
from PIL import Image
import app
image_path = Path('..') / 'GuTelligence-StoMy-Clean-Split' / 'test' / 'Type 1' / 'test__type1_1_jpg.rf.a203c074ce6e75b9266fe03af3169bbe.jpg'
img = Image.open(image_path)
result = app.predict(img)
print(result[0])
print(result[1])
print(result[2])
print(result[3])
print(result[4]['registry_version'])
'@ | ..\.venv\Scripts\python.exe -
```

Smoke-test output:

```text
Type 1/2 hard
0.4168
0.085
{'Type 1/2 hard': 0.41677337884902954, 'Type 3/4 normal-range': 0.14977066218852997, 'Type 5/6/7 loose-watery': 0.4334560036659241}
2026-06-09-product-3class-v1
```

Deployment blocker:

- Hugging Face login succeeded as `perram27`.
- Creating the Space under `GuTelligence-Limited` failed with `403 Forbidden` because the token/user did not have organization write permission.
- Created MVP Space under the logged-in user namespace:

```text
https://huggingface.co/spaces/perram27/bsfs-3class-type7-risk-mvp
```

- `git push` to Hugging Face timed out when connecting to `huggingface.co:443`.
- `hf upload` succeeded for small files:

```text
.gitattributes
README.md
app.py
model_registry.json
requirements.txt
```

- Uploading `checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth` timed out after repeated attempts from the current network.
- Current Space state: code is deployed, but inference cannot run until the checkpoint is uploaded.
- Next action: retry checkpoint upload from a stable network or upload the checkpoint through the Hugging Face web UI to the exact registry path.

### Type 7 Risk Calibration on Current Primary Model

Date: 2026-06-08

Goal:

- Calibrate Type 7 watery-risk thresholds for the current primary ConvNeXt-Tiny model.
- Avoid selecting thresholds directly on test by exporting full-probability validation and test predictions, selecting on validation, and reporting on test.

Files added:

```text
calibrate_type7_risk.py
type7_risk_calibration/valid_predictions_full_probs.csv
type7_risk_calibration/test_predictions_full_probs.csv
type7_risk_calibration/valid_grouped_3class_type7_risk.md
type7_risk_calibration/test_grouped_3class_type7_risk.md
type7_risk_calibration/type7_calibration.md
type7_risk_calibration/type7_calibration.json
```

Validation split metrics from full 7-class probabilities:

```text
3-class grouped accuracy : 79.44%
3-class grouped macro F1 : 0.7694
Type 7 ROC-AUC           : 0.9649
Type 7 average precision : 0.6731
Type 7 support           : 19
```

Test split metrics from full 7-class probabilities:

```text
3-class grouped accuracy : 79.63%
3-class grouped macro F1 : 0.7946
Type 7 ROC-AUC           : 0.6815
Type 7 average precision : 0.2214
Type 7 support           : 19
```

Threshold selection:

```text
Policy: choose the highest-precision threshold subject to validation recall >= 0.50
Selected threshold: 0.44

Validation at 0.44:
  precision 0.7368 | recall 0.7368 | FNR 0.2632 | FPR 0.0256 | flagged 0.0888

Test at 0.44:
  precision 0.3636 | recall 0.2105 | FNR 0.7895 | FPR 0.0355 | flagged 0.0509
```

Interpretation:

- The validation split makes Type 7 risk look highly separable, but the test split does not confirm that separation.
- The selected validation threshold `0.44` does not generalize to test; test Type 7 recall remains only `0.2105`.
- This is evidence that Type 7 risk thresholding is unstable with the current small public-style split, especially with only `19` Type 7 samples per validation/test split.

Decision:

- Do not add a fixed Type 7 risk threshold to `model_registry.json` yet.
- Keep Type 7 probability exposed as a continuous risk score.
- Treat Type 7 risk calibration as blocked on repeated-split calibration and, preferably, device-native validation data.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile calibrate_type7_risk.py evaluate_3class_type7_risk.py bsfs_inference.py run_repeated_split_experiment.py
.\.venv\Scripts\python.exe calibrate_type7_risk.py --min-recall 0.50
```

### Product 3-Class Inference Schema

Date: 2026-06-08

Goal:

- Update the single-image inference wrapper to expose the formal product-facing 3-class output schema.
- Preserve existing 7-class, clinical, and 4-class fields for backward compatibility.

Files updated:

```text
bsfs_inference.py
INFERENCE.md
type7_risk_calibration/sample_product_schema_output.json
```

New output fields:

```text
product_schema
primary_group
primary_group_confidence
bsfs_continuous_score
product_group_score
type7_probability
```

`product_schema` now includes:

```text
schema_version: bsfs_product_3class_v1
primary_group: Type 1/2 hard | Type 3/4 normal-range | Type 5/6/7 loose-watery
primary_group_confidence
group_probabilities
probability_argmax_group
bsfs_continuous_score
product_group_score
type7_probability
type7_risk.threshold_status: not_calibrated_for_fixed_flag
raw_top1_label
raw_top1_confidence
raw_probabilities
```

Design decision:

- `primary_group` uses the raw 7-class top-1 label mapped into the 3-class product group, matching the measured `79.63%` clean-test 3-class metric.
- `probability_argmax_group` is also exposed because group probability sums can disagree with raw-top1 mapping.
- `type7_probability` is exposed as a continuous risk score only. No fixed watery-risk flag is emitted because validation-selected thresholds did not generalize to test.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile bsfs_inference.py
.\.venv\Scripts\python.exe bsfs_inference.py --image ".\GuTelligence-StoMy-Clean-Split\test\Type 1\test__type1_1_jpg.rf.a203c074ce6e75b9266fe03af3169bbe.jpg" --mode best_accuracy --output .\type7_risk_calibration\sample_product_schema_output.json
```

### Repository Cleanup for Product 3-Class Scope

Date: 2026-06-09

Goal:

- Reduce the repository to files relevant to the current deployment direction:
  3-class product schema with Type 7 retained as a continuous risk probability.
- Remove rejected historical experiment scripts and obsolete checkpoint directories.

Kept Python files:

```text
bsfs_inference.py
calibrate_type7_risk.py
evaluate_3class_type7_risk.py
model.py
train_convnext_tiny_3class_type7_aux.py
```

Kept active directories:

```text
checkpoints_clean_split_convnext_tiny/
GuTelligence-StoMy-Clean-Split/
GuTelligence-StoMy-First-Project-2/
type7_risk_calibration/
```

Deleted rejected or historical Python scripts:

```text
analyze_errors.py
analyze_grouped_errors.py
audit_dataset.py
create_clean_split.py
create_repeated_clean_splits.py
evaluate.py
evaluate_ensemble.py
evaluate_grouped_metrics.py
evaluate_grouped_mixed_ensemble.py
evaluate_grouped_selective.py
evaluate_hierarchical.py
evaluate_mixed_ensemble.py
evaluate_relaxed_metrics.py
evaluate_roi_preprocessing.py
evaluate_selective.py
evaluate_type7_cascade.py
evaluate_type7_gate.py
fine_tune_focal.py
fine_tune_highres.py
fine_tune_ordinal_soft.py
fine_tune_type3.py
gpu.py
materialize_clean_split_from_manifest.py
optimize_ensemble.py
optimize_single_postprocess.py
review_type3_type4.py
run_repeated_split_experiment.py
summarize_repeated_splits.py
train_convnext_tiny.py
train_convnext_tiny_4class_grouped.py
train_convnext_tiny_multitask.py
train_convnext_tiny_ordinal.py
train_convnext_tiny_type34_merged.py
train_neighbor_classifier.py
train_type7_binary.py
training.py
```

Deleted obsolete directories:

```text
checkpoints/
checkpoints_clean_split_convnext_small/
checkpoints_clean_split_efficientnet_v2_s/
checkpoints_clean_split_tuned_300/
checkpoints_clean_split_tuned_300_macro_f1/
checkpoints_clean_split_focal_ft_tuned300/
dataset_audit/
error_analysis_grouped_4class/
repeated_clean_splits/
repeated_split_experiments/
selective_results/
type3_type4_review_artifacts/
__pycache__/
```

Registry and docs updated:

```text
model_registry.json
README.md
INFERENCE.md
```

Registry status:

- Active mode: `product_3class`.
- Backward-compatible alias: `best_accuracy`.
- Removed old `high_coverage` and `cautious` EfficientNet modes.
- Registry version: `2026-06-09-product-3class-v1`.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile bsfs_inference.py calibrate_type7_risk.py evaluate_3class_type7_risk.py model.py train_convnext_tiny_3class_type7_aux.py
.\.venv\Scripts\python.exe bsfs_inference.py --image ".\GuTelligence-StoMy-Clean-Split\test\Type 1\test__type1_1_jpg.rf.a203c074ce6e75b9266fe03af3169bbe.jpg" --output .\type7_risk_calibration\sample_product_schema_output_after_repo_cleanup.json
```

Cleanup result:

```text
Remaining root Python files: 5
No Python training process remained running.
No temporary split directories remained.
```

### CTO Model Progress Brief

Date: 2026-06-05

Goal:

- Prepare an English LaTeX summary for CTO communication.
- Summarize current best model performance, rejected research directions, and the proposed 3-class grouped product direction with Type 7 probability retained as a separate risk signal.

File added:

```text
CTO_MODEL_PROGRESS_BRIEF.tex
```

Key points documented:

- Current best model: ConvNeXt-Tiny.
- Best single clean-test 3-class grouped accuracy: `79.63%`.
- Conservative repeated split 3-class grouped accuracy mean: `72.38%`.
- Larger backbones and multi-task loss tuning have not improved robust repeated-split metrics.
- Recommended next direction: formal 3-class evaluation, confidence calibration, Type 7 risk metrics, and device-native validation.

### Formal 3-Class Evaluation and Type 7 Risk Metrics

Date: 2026-06-08

Goal:

- Establish formal 3-class product metrics and Type 7 watery-risk metrics.
- Support both current 7-class post-hoc predictions and future models that directly output 3-class group probabilities plus Type 7 probability.

Files added or updated:

```text
evaluate_3class_type7_risk.py
run_repeated_split_experiment.py
checkpoints_clean_split_convnext_tiny/grouped_3class_type7_risk.md
checkpoints_clean_split_convnext_tiny/grouped_3class_type7_risk.json
```

Metrics added:

- 3-class grouped accuracy.
- 3-class grouped macro F1.
- 3-class grouped weighted F1.
- Type 7 ROC-AUC.
- Type 7 average precision.
- Type 7 threshold metrics at `0.10`, `0.20`, `0.30`, `0.40`, and `0.50`: precision, recall, false negative rate, false positive rate, and flagged rate.

Current best ConvNeXt-Tiny post-hoc baseline on the single clean test:

```text
3-class grouped accuracy : 79.63%
3-class grouped macro F1 : 0.7946
3-class weighted F1      : 0.7974
Type 7 ROC-AUC           : 0.7457
Type 7 average precision : 0.2306
```

Type 7 risk threshold examples for the current post-hoc baseline:

```text
threshold 0.10: precision 0.2667 | recall 0.4211 | FNR 0.5789 | flagged 0.1389
threshold 0.20: precision 0.3684 | recall 0.3684 | FNR 0.6316 | flagged 0.0880
threshold 0.30: precision 0.3571 | recall 0.2632 | FNR 0.7368 | flagged 0.0648
threshold 0.50: precision 0.3636 | recall 0.2105 | FNR 0.7895 | flagged 0.0509
```

Interpretation:

- The formal 3-class grouped metric confirms that the current best model is much more product-aligned under `Type 1/2`, `Type 3/4`, and `Type 5/6/7`.
- Existing Type 7 risk estimates from the current 7-class CSV are conservative because historical prediction CSVs only contain top-3 probabilities, not full 7-class probabilities.
- Future risk evaluation should prefer explicit `prob_type7` outputs.

### ConvNeXt-Tiny 3-Class + Type 7 Auxiliary Model

Date: 2026-06-08

Goal:

- Train a product-aligned model with a 3-class primary head and a Type 7 binary risk head.
- Compare it against the current best 7-class ConvNeXt-Tiny model mapped post-hoc into 3 classes.

Files added:

```text
train_convnext_tiny_3class_type7_aux.py
repeated_split_experiments/smoke_convnext_tiny_3class_type7_aux/
repeated_split_experiments/convnext_tiny_3class_type7_aux_light_3split/
```

Model:

```text
Backbone: ConvNeXt-Tiny
Primary head: 3 product groups
Auxiliary head: Type 7 binary risk
Loss: group CE + 0.35 * Type 7 BCE
Schedule: head 4 epochs, full 8 epochs, patience 4
Seeds: 20260601, 20260602, 20260603
```

Repeated split results:

```text
3-class grouped accuracy  mean 0.6914 | std 0.0495 | min 0.6481 | max 0.7454
3-class grouped macro F1  mean 0.6056 | std 0.0596 | min 0.5563 | max 0.6718
Type 7 ROC-AUC            mean 0.7456 | std 0.1148 | min 0.6447 | max 0.8704
Type 7 average precision  mean 0.2660 | std 0.0568 | min 0.2232 | max 0.3304
Type 7 recall at 0.50     mean 0.0000
```

Comparison to current 7-class ConvNeXt-Tiny post-hoc repeated baseline:

```text
3-class grouped accuracy mean:
  3-class + Type7 auxiliary: 0.6914
  7-class post-hoc baseline: 0.7238

3-class grouped macro F1 mean:
  3-class + Type7 auxiliary: 0.6056
  7-class post-hoc baseline: 0.6672
```

Interpretation:

- The direct 3-class + Type 7 auxiliary model did not beat the current 7-class ConvNeXt-Tiny post-hoc 3-class baseline.
- The Type 7 risk head provides usable ranking signal by ROC-AUC, but the default `0.50` threshold is too high and produces zero recall in this short repeated-split run.
- The current recommended product baseline remains the 7-class ConvNeXt-Tiny model mapped post-hoc to 3 classes, with Type 7 probability/risk evaluated separately.

Decision:

- Do not update `model_registry.json`.
- Keep `checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth` as the primary model.
- Continue Type 7 risk work as a calibration/thresholding problem before retraining another architecture.

Verification and cleanup:

```powershell
.\.venv\Scripts\python.exe -m py_compile evaluate_3class_type7_risk.py run_repeated_split_experiment.py train_convnext_tiny_3class_type7_aux.py
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name smoke_convnext_tiny_3class_type7_aux --train-script train_convnext_tiny_3class_type7_aux.py --seeds 20260601 --head-epochs 1 --full-epochs 1 --patience 1
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name convnext_tiny_3class_type7_aux_light_3split --train-script train_convnext_tiny_3class_type7_aux.py --seeds 20260601 20260602 20260603 --head-epochs 4 --full-epochs 8 --patience 4
```

```text
No temporary split directories remained.
No .pth files remained in the smoke or repeated experiment directories.
No Python training process remained running.
```

### ConvNeXt-Small Repeated Split Backbone Experiment

Date: 2026-06-05

Goal:

- Test whether increasing the ConvNeXt backbone size improves robust clean-split metrics over the current ConvNeXt-Tiny baseline.
- Use the same repeated split protocol and lightweight schedule, with a smaller batch size to reduce memory risk.

Experiment:

```text
Backbone: ConvNeXt-Small
Pretraining: torchvision ConvNeXt_Small_Weights.IMAGENET1K_V1
Classifier output: 7 BSFS classes
Parameter count: 49,460,071
Schedule: head 4 epochs, full 8 epochs, patience 4
Batch size: 12
Seeds: 20260601, 20260602, 20260603
```

Repeated split results:

```text
Strict accuracy           mean 0.4120 | std 0.0791 | min 0.3287 | max 0.4861
Macro F1                  mean 0.3609 | std 0.0738 | min 0.2857 | max 0.4333
Type 3/4 relaxed accuracy mean 0.4784 | std 0.0675 | min 0.4074 | max 0.5417
4-class grouped accuracy  mean 0.5972 | std 0.0501 | min 0.5556 | max 0.6528
4-class grouped macro F1  mean 0.5315 | std 0.0691 | min 0.4846 | max 0.6109
Adjacent accuracy         mean 0.7485 | std 0.0552 | min 0.7037 | max 0.8102
MAE                       mean 1.1389 | std 0.1837 | min 0.9306 | max 1.2778
Type 7 recall             mean 0.4211 | std 0.1823 | min 0.3158 | max 0.6316
Type 7 precision          mean 0.3315 | std 0.0903 | min 0.2500 | max 0.4286
```

Comparison to lightweight ConvNeXt-Tiny repeated baseline:

```text
Grouped accuracy          0.5972 vs 0.6420
Strict accuracy           0.4120 vs 0.4645
Macro F1                  0.3609 vs 0.4188
Type 3/4 relaxed accuracy 0.4784 vs 0.5139
Adjacent accuracy         0.7485 vs 0.7917
Type 7 recall             0.4211 vs 0.5263
Type 7 precision          0.3315 vs 0.4203
```

Interpretation:

- ConvNeXt-Small did not improve any primary metric over ConvNeXt-Tiny.
- The larger parameter count did not translate into better generalization on the current dataset.
- This supports the hypothesis that the current bottleneck is data distribution, label ambiguity, and product validation mismatch rather than model capacity.

Decision:

- Reject ConvNeXt-Small as a replacement candidate under the current dataset and schedule.
- Do not update `model_registry.json`.
- Stop architecture-only escalation for now. Further gains are more likely to come from device-native data, confidence calibration, OOD handling, event-level aggregation, or label policy changes than from larger backbones.

Verification and cleanup:

```powershell
.\.venv\Scripts\python.exe -m py_compile train_convnext_tiny.py run_repeated_split_experiment.py materialize_clean_split_from_manifest.py
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name smoke_convnext_small_repeated --train-script train_convnext_tiny.py --seeds 20260601 --head-epochs 1 --full-epochs 1 --patience 1 --batch-size 12
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name convnext_small_light_3split --train-script train_convnext_tiny.py --seeds 20260601 20260602 20260603 --head-epochs 4 --full-epochs 8 --patience 4 --batch-size 12
```

```text
No temporary split directories remained.
No .pth files remained in the smoke or repeated experiment directories.
No Python training process remained running.
```

### EfficientNetV2-S Repeated Split Backbone Experiment

Date: 2026-06-05

Goal:

- Test a stronger but still deployment-feasible backbone after multi-task loss tuning failed to improve robust metrics.
- Use the same repeated clean split protocol and lightweight schedule as the ConvNeXt-Tiny baseline.

Files updated:

```text
train_convnext_tiny.py
repeated_split_experiments/smoke_efficientnet_v2_s_repeated/
repeated_split_experiments/efficientnet_v2_s_light_3split/
```

Code changes:

- Added `BSFS_BACKBONE` to `train_convnext_tiny.py`.
- Supported backbones:
  - `convnext_tiny` default, preserving historical behavior.
  - `convnext_small`.
  - `efficientnet_v2_s`.
- Default checkpoint directory now follows the selected backbone unless `BSFS_CHECKPOINT_DIR` is explicitly set.

Model details:

```text
Backbone: EfficientNetV2-S
Pretraining: torchvision EfficientNet_V2_S_Weights.IMAGENET1K_V1
Classifier output: 7 BSFS classes
Parameter count: 20,186,455
Schedule: head 4 epochs, full 8 epochs, patience 4
Seeds: 20260601, 20260602, 20260603
```

Repeated split results:

```text
Strict accuracy           mean 0.3935 | std 0.0807 | min 0.3009 | max 0.4491
Macro F1                  mean 0.3549 | std 0.0784 | min 0.2645 | max 0.4047
Type 3/4 relaxed accuracy mean 0.4398 | std 0.0490 | min 0.3843 | max 0.4769
4-class grouped accuracy  mean 0.5756 | std 0.0499 | min 0.5185 | max 0.6111
4-class grouped macro F1  mean 0.5123 | std 0.0686 | min 0.4428 | max 0.5800
Adjacent accuracy         mean 0.6898 | std 0.0350 | min 0.6574 | max 0.7269
MAE                       mean 1.2207 | std 0.0932 | min 1.1343 | max 1.3194
Type 7 recall             mean 0.2807 | std 0.2127 | min 0.1579 | max 0.5263
Type 7 precision          mean 0.4185 | std 0.1913 | min 0.2000 | max 0.5556
```

Comparison to lightweight ConvNeXt-Tiny repeated baseline:

```text
Grouped accuracy          0.5756 vs 0.6420
Strict accuracy           0.3935 vs 0.4645
Macro F1                  0.3549 vs 0.4188
Type 3/4 relaxed accuracy 0.4398 vs 0.5139
Adjacent accuracy         0.6898 vs 0.7917
Type 7 recall             0.2807 vs 0.5263
Type 7 precision          0.4185 vs 0.4203
```

Interpretation:

- EfficientNetV2-S did not improve any primary robust metric over ConvNeXt-Tiny.
- Type 7 recall was substantially worse, which is unacceptable for the current product grouping weakness.
- This backbone is not a promising replacement under the current schedule and dataset.

Decision:

- Reject EfficientNetV2-S as a replacement candidate.
- Do not update `model_registry.json`.
- Continue stronger-backbone search with ConvNeXt-Small before shifting fully to calibration/OOD/product-confidence work.

Verification and cleanup:

```powershell
.\.venv\Scripts\python.exe -m py_compile train_convnext_tiny.py run_repeated_split_experiment.py materialize_clean_split_from_manifest.py
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name smoke_efficientnet_v2_s_repeated --train-script train_convnext_tiny.py --seeds 20260601 --head-epochs 1 --full-epochs 1 --patience 1
.\.venv\Scripts\python.exe run_repeated_split_experiment.py --name efficientnet_v2_s_light_3split --train-script train_convnext_tiny.py --seeds 20260601 20260602 20260603 --head-epochs 4 --full-epochs 8 --patience 4
```

```text
No temporary split directories remained.
No .pth files remained in the smoke or repeated experiment directories.
No Python training process remained running.
```
