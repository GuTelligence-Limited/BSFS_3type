# BSFS Model Development Plan

This plan defines the development phases for the GuTelligence StoMy BSFS image-classification model. Each phase has a clear goal, required work, acceptance criteria, and output artifacts.

The immediate priority is to turn the current prototype into a reproducible, trustworthy, product-relevant model pipeline.

## Change Logging Rule

All code, dataset-layout, report, documentation, and model-workflow changes must be recorded in `DEVELOPMENT_LOG.md`.

When a change affects phase status, acceptance criteria, current risks, or next actions, update this plan in the same work session.

Minimum log entry content:

- date
- files changed
- summary of the change
- reason for the change
- verification or commands run, when applicable
- key metrics or audit results, when applicable

## Phase 0 - Baseline Audit and Reproducibility

Goal: make the current prototype reproducible and establish a reliable baseline.

Tasks:

- Freeze the current dataset snapshot and record exact file counts per split and class.
- Verify whether Roboflow-exported augmented or duplicate images appear across train, validation, and test.
- Regenerate evaluation figures from the latest EfficientNet-B4 checkpoint.
- Confirm exact commands required to run training and evaluation on the development machine.
- Record environment versions for Python, PyTorch, torchvision, CUDA, and GPU.
- Add a lightweight experiment log for model version, dataset version, training date, checkpoint path, and metrics.

Acceptance criteria:

- One command can regenerate test metrics from a saved checkpoint.
- The latest figures and `metrics_summary.txt` match `training_history.json`.
- Any known data leakage or duplicate risk is documented.
- Baseline metrics are treated as provisional until leakage checks are complete.

Artifacts:

- updated `README.md`
- refreshed `figures/`
- `figures/metrics_summary.txt`
- dataset split summary
- baseline experiment record

## Phase 1 - Dataset Integrity and Label Quality

Goal: improve trust in labels and split quality before tuning model architecture further.

Status on 2026-06-01: started. A first-pass read-only audit script has been added as `audit_dataset.py`, with reports written to `dataset_audit/`.

Tasks:

- Detect exact duplicate images by file hash.
- Detect near-duplicates using image embeddings or perceptual hashes.
- Ensure related images from the same source do not cross train/valid/test boundaries.
- Review mislabeled and ambiguous samples, especially Type 3/4/5/6/7.
- Define a label policy for borderline BSFS cases.
- Decide whether the problem should remain 7-class classification or include an ordinal/near-miss evaluation.
- Rebuild a clean split after duplicate and label review.

Acceptance criteria:

- Clean split has no known duplicate leakage.
- Label-review notes exist for ambiguous examples.
- Each class has enough representative validation and test examples for metric reporting.
- A new baseline is trained and compared against the current model.

Artifacts:

- cleaned dataset manifest
- duplicate/near-duplicate report
- label policy document
- clean-split baseline checkpoint
- clean-split evaluation report

Initial audit findings:

- Current local dataset contains 2149 image files.
- Exact SHA-256 duplicate groups: 137.
- Exact duplicate groups crossing train/valid/test splits: 72.
- Source-name duplicate groups after stripping Roboflow `.rf.<hash>` suffixes: 238.
- Source-name duplicate groups crossing train/valid/test splits: 197.
- aHash near-duplicate pairs at Hamming distance <= 6: 1950.
- aHash near-duplicate pairs crossing splits: 873.
- aHash near-duplicate pairs with label conflicts: 724.

Immediate Phase 1 action:

- Treat the original train/valid/test split as contaminated for final model reporting.
- Use `GuTelligence-StoMy-Clean-Split/` as the current cleaned split.
- Manually review label-conflict near-duplicate pairs before using them as ground truth errors.
- Re-train and re-evaluate the current EfficientNet-B4 baseline only after the cleaned split is created.

Clean split status:

- Script: `create_clean_split.py`.
- Output: `GuTelligence-StoMy-Clean-Split/`.
- Total copied images: 2149.
- Connected components assigned: 299.
- Split counts: train 1719, valid 214, test 216.
- `source_id` cross-split groups: 0.
- `sha256` cross-split groups: 0.

Clean split EfficientNet-B4 baseline:

- Checkpoint directory: `checkpoints_clean_split/`.
- Main figures directory: `figures_clean_split/`.
- Best-full checkpoint figures directory: `figures_clean_split_best_full/`.
- Final checkpoint fixed-seed TTA test accuracy: 37.04%.
- Final checkpoint fixed-seed TTA macro F1: 0.3815.
- Final checkpoint fixed-seed TTA macro ROC-AUC: 0.7646.
- Training final evaluation without TTA: 32.87% accuracy, 0.3332 macro F1.
- Best full fine-tune checkpoint TTA test accuracy: 37.50%.
- Best full fine-tune checkpoint TTA macro F1: 0.3966.

Clean split error analysis:

- Script: `analyze_errors.py`.
- Output directory: `error_analysis_clean_split/`.
- Manual review file: `error_analysis_clean_split/failure_review.csv`.
- Total test images: 216.
- Errors: 136.
- Adjacent accuracy (+/-1): 72.69%.
- Ordinal MAE: 1.0370.
- High-confidence errors (confidence >= 0.50): 72.
- Most common error pairs: Type 6 -> Type 5, Type 6 -> Type 4, Type 4 -> Type 5, Type 3 -> Type 4, Type 5 -> Type 4, Type 7 -> Type 5.

Phase 1 implication:

- The original 81.215% test accuracy should be treated as inflated by split contamination.
- Current clean split baseline is not product-ready.
- Prioritize label review and data-quality work for Type 3, Type 5, Type 6, and Type 7 before deeper architecture tuning.
- Start manual review from high-confidence errors and the Type 3/5/6/7 error folders in `error_analysis_clean_split/`.

Next clean-split training profile:

- Use the current cleaned split labels as ground truth for the next run.
- Input size: 300.
- Resize size: 340.
- Batch size: 20.
- Disable Mixup.
- Remove vertical flip, Gaussian blur, and random erasing.
- Use lighter crop, rotation, and color jitter.
- Reduce label smoothing to 0.02.
- Lower fine-tuning learning rates.
- Store outputs in a new experiment directory, such as `checkpoints_clean_split_tuned_300/` and `figures_clean_split_tuned_300/`.

Tuned 300 result:

- Checkpoint directory: `checkpoints_clean_split_tuned_300/`.
- Figures directory: `figures_clean_split_tuned_300/`.
- Error analysis directory: `error_analysis_clean_split_tuned_300/`.
- Fixed-seed TTA accuracy: 42.13%.
- Fixed-seed TTA macro F1: 0.3878.
- Fixed-seed TTA macro ROC-AUC: 0.7688.
- Adjacent accuracy (+/-1): 77.31%.
- Ordinal MAE: 0.9306.
- Main improvement over previous clean-split baseline: exact accuracy and adjacent accuracy.
- Main regression/risk: Type 3 recall collapsed to 0.00; Type 6 remains weak.

Next optimization direction:

- Add class-balanced or focal-style loss to improve weak-class recall.
- Track ordinal metrics during validation.
- Consider an ordinal-aware auxiliary loss or post-processing strategy.
- Keep 300x300 input and morphology-preserving augmentation unless memory or runtime becomes a blocker.

Prepared next experiment:

- Checkpoint selection: validation macro F1 instead of validation accuracy.
- Early stopping: validation macro F1.
- Loss: `CrossEntropyLoss` with square-root inverse-frequency class weights and label smoothing 0.02.
- Additional validation metrics in history: macro F1, adjacent accuracy, ordinal MAE.
- Suggested output directories:
  - `checkpoints_clean_split_tuned_300_macro_f1/`
  - `figures_clean_split_tuned_300_macro_f1/`
  - `error_analysis_clean_split_tuned_300_macro_f1/`

Follow-up results on 2026-06-01:

- Evaluation TTA was changed from random crop views to deterministic 5-view TTA.
- `training.py` and `evaluate.py` now default to `GuTelligence-StoMy-Clean-Split/` to prevent accidental reporting on the original contaminated split.
- `tuned_300_macro_f1` improved macro F1 and ordinal metrics but reduced exact accuracy versus `tuned_300`.
- `tuned_300_composite` did not beat the `tuned_300` single-model baseline.
- A weighted ensemble of `tuned_300` and `tuned_300_macro_f1` is the current best measured candidate.

Current clean-split deterministic TTA results:

```text
convnext_tiny final:
  Accuracy: 48.15%
  Macro F1: 0.4727
  Adjacent accuracy: 86.57%
  Ordinal MAE: 0.7130

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

highres_380_tuned300:
  Accuracy: 37.04%
  Macro F1: 0.3605
  Adjacent accuracy: 75.93%
  Ordinal MAE: not primary; see figures_clean_split_highres_380_tuned300/
```

Current candidate status:

- Best single model and best measured full-coverage candidate: `checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth`.
- Best EfficientNet-only historical candidate: `ensemble_results/tuned300_macro_f1_80_20/`.
- The ConvNeXt candidate should be considered a research model until latency, memory, export format, and product-mode abstention are reviewed.
- Mixed ConvNeXt + EfficientNet ensembles were tested and rejected as primary candidates because they did not beat the ConvNeXt single model.

Updated next optimization direction:

- Stop treating class frequency alone as the root cause; class-weighted loss did not solve Type 3.
- Prioritize Type 3/4 boundary work when manual review is available; until then, continue model-only architecture tests only when they can be cleanly compared and do not overwrite current candidates.
- Keep deterministic TTA for all reported comparisons.
- Do not report metrics from the original 181-image split except as historical contaminated baseline.

Autonomous iteration update on 2026-06-01:

- Validation-tuned ensemble postprocessing did not generalize to test and is rejected.
- Focal-loss fine-tune improved single-model macro F1 but did not improve exact accuracy or Type 3 recall.
- Type 3 oversampling improved validation Type 3 recall but failed on test, suggesting validation/test Type 3 distribution mismatch or label ambiguity.
- Type 3/4 diagnostic artifacts were generated:
  - `type3_diagnostic_summary.md`
  - `type3_type4_review.csv`
  - `error_analysis_clean_split_focal_ft_tuned300/`
  - `error_analysis_clean_split_type3_ft_tuned300/`

Updated candidate table:

```text
Best full-coverage exact accuracy:
  checkpoints_clean_split_convnext_tiny/bsfs_convnext_tiny_final.pth
  Accuracy: 48.15%
  Macro F1: 0.4727
  Adjacent accuracy: 86.57%

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

Phase 1 decision:

- Continue to treat Type 3 as a data-quality and boundary-definition blocker.
- Before more architecture tuning, review Type 3/4 examples in `type3_type4_review.csv`.
- If labels are confirmed, test a dedicated Type 3-vs-Type 4/neighbor binary auxiliary model or an ordinal-regression head in Phase 3.

Phase 3 architecture update:

- High-resolution EfficientNet-B4 fine-tuning at 380x380 was tested and rejected because test TTA accuracy fell to `37.04%`.
- ConvNeXt-Tiny improved the current clean-test full-coverage result to `48.15%` exact accuracy and `0.4727` macro F1.
- ConvNeXt-Tiny also improved adjacent accuracy to `86.57%`, which is materially better for an ordinal BSFS product signal.
- Type 3 recall improved from repeated `0.00` EfficientNet results to `0.1739`, but this remains far below product-grade performance.
- ConvNeXt + EfficientNet mixed ensembles were validation-selected and evaluated once on test; neither exceeded the ConvNeXt single model.
- ConvNeXt validation-selected temperature/class-bias postprocessing improved macro F1 slightly (`0.4793`) but lowered exact accuracy (`47.69%`) and adjacent accuracy (`83.33%`), so it is not the primary candidate.
- A direct 6-class ConvNeXt model with Type 3/4 merged into `Type 3/4 normal-range` was trained. It reached `54.63%` single-view accuracy and `52.78%` TTA accuracy, which did not beat the current 7-class ConvNeXt plus Type 3/4 relaxed product rule (`57.41%`).
- A 4-class product grouping was evaluated: Type 1/2 hard, Type 3/4 normal-range, Type 5/6 soft-loose, and Type 7 watery. Post-hoc grouping of the current 7-class ConvNeXt reached `70.37%` grouped accuracy and `0.6292` grouped macro F1.
- A direct 4-class ConvNeXt was trained but reached only `64.81%` test accuracy, so the current recommended product strategy is still the 7-class ConvNeXt plus grouped post-processing.
- 4-class grouped selective prediction was evaluated and rejected. Validation selected `confidence >= 0.25` and `margin >= 0.12`, but test accuracy on kept samples was only `69.61%` at `94.44%` coverage.
- A Type 7 probability gate was evaluated and rejected. It improved validation Type 7 recall but lowered test grouped accuracy to `66.67%`.
- A Type 5/6 vs Type 7 binary specialist was trained. The subtask reached `82.86%` TTA accuracy, but Type 7 recall was only `42.11%`.
- A Type 7 cascade using the binary specialist was evaluated and rejected. It lowered test grouped accuracy to `65.28%` and Type 7 precision to `23.33%`.
- Inference grouping was aligned to the current best measured product metric: raw 7-class top-1 mapped into 4 BSFS product groups. This remains `70.37%` on clean test; group-sum-argmax is lower at `68.98%`.
- Automatic ROI preprocessing was evaluated without retraining and rejected. The best validation strategy remained `standard_center`; ROI crops did not beat standard preprocessing.
- A stronger-regularized ConvNeXt seed2 was trained and rejected. Test TTA strict accuracy fell to `40.74%`.
- A baseline-hyperparameter ConvNeXt seed3 was trained. It reached `47.69%` strict accuracy and `0.4817` macro F1, but product grouped accuracy was `68.52%`, below the current `70.37%` baseline.
- A grouped ensemble of the current best ConvNeXt and seed3 was valid-selected and rejected. It reached `79.44%` grouped accuracy on valid but only `67.59%` on test.
- Repeated leakage-safe clean split manifests were generated for seeds `20260601` through `20260605`. All five splits have zero source/SHA leakage, identical class counts, and 12.94%-17.96% pairwise assignment diversity.
- `GuTelligence-StoMy-Clean-Split-Seed20260602/` was materialized successfully as a dry-run ImageFolder dataset for future repeated-split training.
- A repeated-split lightweight ConvNeXt experiment was run on seeds `20260601`, `20260602`, and `20260603` with head 4 epochs and full 8 epochs. It produced grouped 4-class accuracy mean `0.6420` with std `0.0586`, below the current primary model's single-split grouped accuracy `0.7037`.
- Type 7 recall was highly unstable across the repeated splits: mean `0.5263`, std `0.3646`, range `0.3158` to `0.9474`.
- A full-schedule repeated ConvNeXt-Tiny baseline was run on the same three seeds with head 8 epochs and full 22 epochs. It produced grouped 4-class accuracy mean `0.6219` with std `0.0813`, strict accuracy mean `0.4244`, and Type 7 recall mean `0.5263` with std `0.3646`.
- Full-schedule repeated-split performance was lower than the lightweight repeated baseline, so simply training ConvNeXt-Tiny longer is not a reliable accuracy-improvement path.
- An ordinal-aware ConvNeXt-Tiny loss was tested on the lightweight repeated-split schedule. It slightly improved grouped accuracy mean from `0.6420` to `0.6435`, strict accuracy mean from `0.4645` to `0.4707`, and Type 7 precision mean from `0.4203` to `0.4380`, but adjacent accuracy and MAE slightly worsened.
- A 4-class product group + 7-class raw BSFS multi-task ConvNeXt-Tiny was tested on the same lightweight repeated-split schedule. It slightly improved grouped accuracy mean from `0.6420` to `0.6466` and reduced grouped accuracy std from `0.0586` to `0.0457`, but strict accuracy dropped to `0.4090`, macro F1 dropped to `0.3571`, and Type 7 recall dropped to `0.4912`.
- The initial multi-task weighting is therefore rejected for registry replacement. Continue this branch only by increasing raw-loss and raw-fusion weights to see whether diagnostic 7-class quality can be recovered while preserving grouped product accuracy.
- A stronger raw-weighted multi-task setting was tested with raw loss weight `0.65` and raw fusion weight `0.55`. It performed worse than both the initial multi-task setting and the lightweight ConvNeXt repeated baseline: grouped accuracy mean `0.6219`, strict accuracy mean `0.3781`, macro F1 mean `0.3194`, and Type 7 recall mean `0.4737`.
- Stronger raw loss plus stronger raw fusion is rejected. The likely issue is not simply too little raw supervision; fusing a group head into raw predictions can damage fine-grained class quality and still fail to stabilize product grouping.
- A raw-primary multi-task setting was tested with raw 7-class inference and the group head used only as auxiliary loss. It reached grouped accuracy mean `0.6327`, strict accuracy mean `0.4028`, macro F1 mean `0.3497`, Type 7 recall mean `0.4912`, and Type 7 precision mean `0.5896`.
- Raw-primary multi-task improved Type 7 precision versus the lightweight baseline but reduced grouped accuracy, strict accuracy, macro F1, Type 3/4 relaxed accuracy, adjacent accuracy, and Type 7 recall. It is rejected for registry replacement.
- Multi-task group-loss tuning has now been tested in three forms and has not produced a robust improvement over plain ConvNeXt-Tiny.
- EfficientNetV2-S was tested as a stronger but deployment-feasible backbone with the same repeated split protocol. It reached grouped accuracy mean `0.5756`, strict accuracy mean `0.3935`, macro F1 mean `0.3549`, adjacent accuracy mean `0.6898`, and Type 7 recall mean `0.2807`, all worse than ConvNeXt-Tiny except effectively tied Type 7 precision.
- EfficientNetV2-S is rejected as a replacement candidate under the current dataset and schedule.
- ConvNeXt-Small was tested as a larger same-family backbone with batch size `12`. It reached grouped accuracy mean `0.5972`, strict accuracy mean `0.4120`, macro F1 mean `0.3609`, adjacent accuracy mean `0.7485`, and Type 7 recall mean `0.4211`, all worse than ConvNeXt-Tiny.
- ConvNeXt-Small is rejected as a replacement candidate under the current dataset and schedule.
- Current evidence suggests the limiting factor is not simply model capacity. Larger backbones and multi-task losses have not produced robust repeated-split gains.

Phase 3 next model-only options while manual relabeling is unavailable:

- Do not expect thresholding, abstention, or a simple Type 7 cascade to reach 90% on the current clean split; these have now been tested and did not generalize.
- For product output, prefer raw 7-class top-1 mapped into the 4-class grouped status. Keep raw 7-class output and group probabilities for diagnostics and confidence transparency.
- Do not train on the current simple automatic ROI heuristic. It did not improve validation metrics over standard center preprocessing.
- Before more valid-selected tuning, improve the validation protocol with repeated clean splits or cross-validation; several valid gains have failed to generalize to test.
- Use `repeated_clean_splits/` manifests for the next model-only experiments. Report mean and standard deviation across at least 3 seeds before accepting a change as real.
- Materialize split manifests with `materialize_clean_split_from_manifest.py`; do not manually copy split folders.
- Use `run_repeated_split_experiment.py` for repeated-split model experiments. Keep `.pth` cleanup enabled unless a model is selected for registry consideration.
- Acceptance threshold for future model-only changes: improve repeated-split grouped 4-class mean and strict/macro F1 mean without materially worsening Type 7 recall variance.
- Next repeated-split experiments should change the training objective, not just train longer. Prioritize ordinal-aware loss and 4-class + 7-class multi-task learning.
- Current ordinal-aware loss is not strong enough for registry consideration. Keep it as an experimental objective and prioritize 4-class + 7-class multi-task learning next.
- Current multi-task loss with raw weight `0.35` is also not strong enough for registry consideration. Next multi-task test should use stronger raw supervision, such as raw loss weight `0.65` and raw fusion weight `0.55`.
- Raw weight `0.65` and fusion weight `0.55` was tested and rejected. If multi-task learning continues, switch to raw-head-only inference with the group head used only as an auxiliary training loss.
- Raw-head-only inference with auxiliary group loss was tested and rejected. Deprioritize more multi-task loss tuning unless there is new data or a materially different validation objective.
- Next model-only architecture work should test a stronger deployment-feasible backbone, such as EfficientNetV2-S or ConvNeXt-Small, using the same repeated split protocol.
- EfficientNetV2-S has been tested and rejected. Continue with ConvNeXt-Small as the next stronger-backbone comparison, then stop architecture-only escalation unless it produces a clear repeated-split improvement.
- ConvNeXt-Small has been tested and rejected. Stop architecture-only escalation for now unless new device-native data changes the validation distribution.
- Next Phase 3/4 work should focus on product confidence behavior: calibration, confidence bins, grouped-risk reporting, OOD/invalid-image handling, and event-level aggregation design.
- Formal 3-class product evaluation was added for `Type 1/2`, `Type 3/4`, and `Type 5/6/7`, with Type 7 retained as a separate watery-risk signal.
- The current best 7-class ConvNeXt-Tiny post-hoc 3-class baseline is `79.63%` on the single clean test, with Type 7 ROC-AUC `0.7457`.
- A direct ConvNeXt-Tiny 3-class primary head plus Type 7 auxiliary risk head was trained on the lightweight repeated split protocol. It reached 3-class accuracy mean `0.6914`, 3-class macro F1 mean `0.6056`, Type 7 ROC-AUC mean `0.7456`, and Type 7 AP mean `0.2660`.
- The direct 3-class + Type 7 auxiliary model is rejected as a replacement because it did not beat the current 7-class ConvNeXt-Tiny post-hoc 3-class repeated baseline of 3-class accuracy mean `0.7238` and macro F1 mean `0.6672`.
- Type 7 risk should now be treated primarily as a calibration and threshold-selection problem. The auxiliary risk head has ranking signal, but the default `0.50` threshold produced zero Type 7 recall in this short repeated-split run.
- Type 7 risk calibration was tested on the current primary ConvNeXt-Tiny model by selecting thresholds on the clean validation split and reporting on test. Validation Type 7 ROC-AUC was `0.9649`, but test ROC-AUC was only `0.6815`, showing strong threshold instability.
- The validation-selected threshold `0.44` reached validation precision/recall `0.7368/0.7368`, but test precision/recall only `0.3636/0.2105`. Do not deploy this as a fixed threshold.
- Product output should expose Type 7 probability as a continuous risk score for now. A categorical watery-risk flag needs repeated-split calibration or device-native validation before registry use.
- `bsfs_inference.py` now emits the formal product schema `bsfs_product_3class_v1`, including `primary_group`, `primary_group_confidence`, `bsfs_continuous_score`, `product_group_score`, and `type7_probability`.
- The inference schema keeps Type 7 as a continuous risk score and explicitly marks fixed threshold status as `not_calibrated_for_fixed_flag`.
- Existing 7-class, Type 3/4 relaxed, and 4-class grouped fields are preserved for backward compatibility.
- The repository was cleaned and re-scoped for deployment/API work. Root Python files were reduced to `bsfs_inference.py`, `calibrate_type7_risk.py`, `evaluate_3class_type7_risk.py`, `model.py`, and `train_convnext_tiny_3class_type7_aux.py`.
- `model_registry.json` was simplified to active mode `product_3class` with `best_accuracy` retained as a backward-compatible alias. Old EfficientNet selective modes were removed.
- Historical rejected experiment scripts, repeated split manifests, old checkpoints, and old analysis directories were removed after their results were preserved in `DEVELOPMENT_LOG.md`.
- Keep the repository lean: retain current registry checkpoints and concise reports, but delete rejected experiment checkpoints after their metrics are recorded in `DEVELOPMENT_LOG.md`.
- Prefer split manifests over persistent duplicate dataset folders; materialized repeated splits should be treated as temporary training inputs.
- Prioritize device-native data collection and a held-out device-native validation set. Without this, validation gains are not reliable enough for product decisions.
- Prioritize additional Type 7 collection because Type 7 remains the weakest product group and the current clean test has only `19` Type 7 samples.
- Consider event-level multi-frame aggregation once smart-seat capture data exists; a single-frame public-image classifier is unlikely to reach the product-level 90% target alone.
- Do not continue stronger-regularized ConvNeXt-Tiny in the tested seed2 profile; it underperformed substantially.
- Keep seed3 as a diagnostic comparison model, not as the primary product model.
- Try EfficientNetV2-S or ConvNeXt-Small only if runtime budget allows; use valid-selected checkpoints and report clean test once.
- Avoid more Type 3-only oversampling unless there is a new validation strategy, because previous oversampling did not generalize to test.

Phase 3 neighbor-classifier experiment update:

- A dedicated Type 2/3/4 classifier initialized from `tuned_300` failed to recover Type 3 on clean test.
- A dedicated Type 2/3/4 classifier initialized from ImageNet recovered some Type 3 recall (`0.2609`) but had poor subtask accuracy (`37.11%`) and too many Type 4 false positives.
- Hierarchical inference using the neighbor classifier did not beat the current single-model or ensemble candidates.

Decision:

- Do not continue model-only Type 3 fixes until `type3_type4_review.csv` is reviewed.
- Treat Type 3/4 definition, label consistency, and data distribution as the Phase 1 blocker.
- Phase 3 architecture work should resume only after labels are corrected or confirmed.

Type 3/4 review package:

- Script: `review_type3_type4.py`.
- Output: `type3_type4_review_artifacts/`.
- Start review from:
  - `type3_type4_review_artifacts/summary.md`
  - `type3_type4_review_artifacts/contact_sheet_top_suspicious.jpg`
  - `type3_type4_review_artifacts/type3_type4_suspicious_ranked.csv`
- Priority source IDs:
  - `type310_jpg`
  - `type314_jpg`
  - `6xjvw61qgtdb1_jpg`
  - `type31_jpg`

Phase 1 next gate:

- Mark each priority row as confirmed, corrected to another BSFS type, or unusable/ambiguous.
- If many Type 3 rows are corrected, rebuild the clean split from corrected labels and retrain `tuned_300`.
- If Type 3 rows are confirmed, collect additional Type 3 examples under a clearer labeling standard before more architecture work.

No-manual-relabel constraint update:

- The team currently cannot manually relabel Type 3/4 data.
- Under this constraint, do not block all model development on manual review.
- Treat Type 3/4 as noisy adjacent-boundary data and prioritize robust ordinal metrics.

No-manual-relabel experiment result:

- Ordinal soft-label fine-tune was tested in `checkpoints_clean_split_ordinal_soft_tuned300/`.
- Result: accuracy `43.98%`, macro F1 `0.4185`, adjacent accuracy `75.93%`, MAE `0.9167`, Type 3 recall `0.00`.
- It did not beat the current best exact candidate or focal single-model macro-F1 candidate.

Temporary operating model until relabeling is possible:

- Best exact-accuracy research candidate: `ensemble_results/tuned300_macro_f1_80_20/`.
- Best single-model candidate for macro F1: `checkpoints_clean_split_focal_ft_tuned300/bsfs_efficientnet_b4_final.pth`.
- Product-facing logic should consider adjacent accuracy, ordinal MAE, and uncertainty around Type 3/4 instead of treating all 7 exact labels as equally reliable.
- Future data collection should prioritize device-native Type 3 and Type 4 examples with clearer capture conditions.

Selective prediction operating modes:

- High-coverage prototype mode:
  - Result: `selective_results/ensemble_80_20_min80/`
  - Model: `tuned_300` 80% + `macro_f1` 20%
  - Coverage: `93.98%`
  - Kept-sample accuracy: `45.32%`
  - Kept-sample adjacent accuracy: `76.85%`
  - Abstains: `13/216`

- Cautious prototype mode:
  - Result: `selective_results/focal_ft_min60/`
  - Model: focal fine-tuned single model
  - Coverage: `70.83%`
  - Kept-sample accuracy: `45.10%`
  - Kept-sample macro F1: `0.4300`
  - Kept-sample adjacent accuracy: `80.39%`
  - Abstains: `63/216`

Product implication:

- Until Type 3/4 labels or device-native data improve, the product should expose uncertainty/insufficient-confidence behavior instead of always returning a hard BSFS type.
- A user-facing system can map abstentions to "unable to classify confidently" or ask for another capture, while keeping internal probability distributions for later review.

Inference prototype:

- Registry: `model_registry.json`.
- Wrapper: `bsfs_inference.py`.
- Documentation: `INFERENCE.md`.
- Supported modes:
  - `high_coverage`
  - `cautious`
- Inference output includes:
  - accepted/rejected status
  - final label when accepted
  - raw top label
  - confidence
  - top1-top2 margin
  - abstention reasons
  - Type 3/4 boundary warning
  - full probability distribution

Deployment note:

- This is a prototype wrapper, not an optimized edge artifact.
- Before Phase 5, benchmark latency/memory and export a runtime artifact such as TorchScript, ONNX, or TensorRT.

## Phase 2 - Real-World Product Data Collection

Goal: collect data that matches the smart toilet seat camera environment.

Tasks:

- Define target camera specifications: resolution, focal length, field of view, lighting, mounting position, and capture timing.
- Collect staged and real-use images under expected device conditions.
- Include variation in lighting, water level, bowl material, occlusion, motion blur, reflections, and partial views.
- Separate public/web-sourced data from device-native data in dataset metadata.
- Define privacy and data-handling rules for sensitive images.
- Build a real-world validation set that is never used for training.

Acceptance criteria:

- Device-native validation set exists and is isolated.
- Each BSFS type has enough examples for at least preliminary per-class evaluation.
- Data collection protocol is repeatable by the engineering team.
- Privacy and retention rules are documented.

Artifacts:

- device data collection protocol
- device-native dataset manifest
- real-world validation split
- privacy and data-handling checklist

## Phase 3 - Model Quality Improvement

Goal: improve generalization and robustness beyond the current EfficientNet-B4 prototype.

Tasks:

- Re-train EfficientNet-B4 on the cleaned dataset and device-native data.
- Compare alternative backbones appropriate for deployment, such as EfficientNetV2, ConvNeXt-Tiny, MobileNetV3, or ViT-small.
- Add ordinal-aware evaluation because BSFS labels have natural ordering.
- Evaluate exact accuracy, macro F1, per-class recall, adjacent-class accuracy, MAE over class index, and calibration error.
- Test class-balanced loss, focal loss, ordinal regression, and two-stage grouping approaches if needed.
- Run ablations for augmentation, Mixup, GeM pooling, TTA, and input resolution.
- Use Grad-CAM and failure-case review to identify shortcut learning.

Acceptance criteria:

- New model improves macro F1 and weak-class recall on clean and device-native validation sets.
- Improvements are reproducible across at least two runs or folds.
- Failure modes are documented with example images.
- The selected model has a clear reason for product continuation.

Artifacts:

- experiment comparison table
- selected candidate checkpoint
- failure-case report
- updated evaluation figures
- model card draft

## Phase 4 - Robustness, Safety, and Clinical Relevance

Goal: evaluate whether the model is reliable enough for health-related user-facing behavior.

Tasks:

- Define medically safe product output language with clinical advisors.
- Separate model prediction from user-facing recommendation logic.
- Evaluate model behavior on out-of-distribution inputs: empty bowl, urine only, toilet paper, blood-like color, cleaning products, poor lighting, and blocked camera.
- Add a reject/uncertain class or confidence threshold if needed.
- Calibrate prediction confidence using validation data.
- Test model stability across multiple captures from the same event.
- Define escalation rules for repeated abnormal classifications.

Acceptance criteria:

- The model can abstain or flag uncertain/invalid images.
- Confidence thresholds are justified by validation metrics.
- Out-of-distribution behavior is measured and documented.
- Product language does not overclaim diagnosis.

Artifacts:

- OOD test set
- calibration report
- uncertainty threshold recommendation
- safety and product-language notes

## Phase 5 - Deployment Optimization

Goal: prepare the selected model for the smart toilet seat runtime environment.

Tasks:

- Define deployment target: edge device, phone, local hub, or cloud.
- Measure latency, memory, model size, and power requirements.
- Export model to the required format, such as TorchScript, ONNX, TensorRT, Core ML, or TFLite.
- Test quantization or distillation if the model is too large.
- Build an inference wrapper with preprocessing identical to training/evaluation.
- Add versioned model metadata: model ID, dataset ID, preprocessing version, class mapping, and confidence thresholds.

Acceptance criteria:

- Inference output matches PyTorch reference within acceptable tolerance.
- Runtime latency and memory fit product constraints.
- Preprocessing is consistent and versioned.
- Model artifact is ready for integration testing.

Artifacts:

- exported model artifact
- inference wrapper
- runtime benchmark report
- model metadata file

## Phase 6 - Product Validation and Monitoring

Goal: validate the model in realistic product workflows and prepare for continuous improvement.

Tasks:

- Run pilot testing with device-native images.
- Track exact-class accuracy, adjacent-class accuracy, invalid-image rate, confidence distribution, and user-impacting error types.
- Build a human review workflow for low-confidence and high-risk predictions.
- Define retraining triggers based on drift, new camera hardware, new lighting conditions, or metric degradation.
- Create model release notes for each deployed model version.
- Establish periodic data and metric review.

Acceptance criteria:

- Pilot validation metrics meet launch criteria agreed by product, engineering, and clinical stakeholders.
- Monitoring plan exists before production deployment.
- Retraining and rollback criteria are documented.
- Model versioning and release process are repeatable.

Artifacts:

- pilot validation report
- monitoring dashboard specification
- retraining protocol
- model release notes

## Working Metrics

Use these metrics consistently across phases:

- exact 7-class accuracy
- Type 3/4 relaxed accuracy, where Type 3 predicted as Type 4 or Type 4 predicted as Type 3 is counted as product-correct
- 4-class grouped product accuracy and macro F1: Type 1/2, Type 3/4, Type 5/6, Type 7
- macro F1
- per-class precision, recall, and F1
- confusion matrix
- adjacent-class accuracy, where prediction within +/-1 BSFS type is counted separately
- ordinal mean absolute error over class index
- calibration error and confidence histogram
- invalid/OOD detection rate
- device-native validation performance

## Current Baseline Targets

Historical contaminated-split baseline, retained only for context:

```text
Test accuracy: 81.215%
Macro F1     : 0.808
```

This is no longer an active target baseline because the earlier split had duplicate/leakage risk. Current clean-split baseline status is:

```text
Best full-coverage exact accuracy: 48.15%  (ConvNeXt-Tiny)
Best Type 3/4 relaxed accuracy   : 57.41%  (ConvNeXt-Tiny)
Best 4-class grouped accuracy    : 70.37%  (7-class ConvNeXt raw top-1 mapped to product groups)
Best full-coverage macro F1      : 0.4727  (ConvNeXt-Tiny)
Best full-coverage adjacent acc. : 86.57%  (ConvNeXt-Tiny)
Best Type 7 cascade accuracy     : 65.28%  (rejected; worse than baseline)
Known blocker                    : Type 7 vs Type 5/6 confusion, limited/noisy labels, small Type 7 test support, and non-device-native data
```

Near-term clean-split model-only targets:

```text
Clean test exact accuracy        >= 55%
Type 3/4 relaxed accuracy        >= 65%
4-class grouped accuracy         >= 75%
Clean test macro F1              >= 0.55
Adjacent-class accuracy          >= 0.88
Type 3 recall                    >= 0.30
```

90% exact accuracy target:

- Treat `90%` as a product-level goal, not a near-term tuning target on the current public/Roboflow-style dataset.
- Track both strict 7-class accuracy and Type 3/4 relaxed accuracy. The relaxed metric is more aligned with the current product interpretation, while strict accuracy remains useful for model diagnostics.
- To make 90% plausible, the project needs device-native data, label policy/review, more examples per class, and a validation set representative of the smart-seat camera.
- The 4-class grouped target is more product-aligned than strict 7-class accuracy, but current best grouped accuracy is still `70.37%`.
- Current evidence rejects a simple confidence-threshold, Type 7 gate, or Type 7 cascade path to 90%.
- Until device-native data, ROI control, and event-level validation exist, model-only work should be judged by clean-test incremental gains and adjacent-class accuracy, not by expecting a jump from ~48% strict or ~70% grouped to 90%.

## Decision Gates

Move from one phase to the next only when the acceptance criteria are met:

- Phase 0 gate: baseline is reproducible.
- Phase 1 gate: clean split and label policy are established.
- Phase 2 gate: device-native validation set exists.
- Phase 3 gate: candidate model improves robust metrics.
- Phase 4 gate: uncertainty, OOD, and safety behavior are acceptable.
- Phase 5 gate: model runs within deployment constraints.
- Phase 6 gate: pilot validation supports product use.

## Repository and Release Policy

Current GitHub repository scope:

- Product-facing 3-class BSFS inference code.
- Continuous Type 7 risk probability and calibration/evaluation scripts.
- Lightweight reports, JSON metrics, documentation, and model registry metadata.

Files intentionally excluded from normal GitHub commits:

- Raw and cleaned image datasets.
- Local IDE and virtual environment files.
- Training logs and Python cache files.
- Model binary artifacts such as `.pth`, `.pt`, `.onnx`, and `.safetensors`.

Release direction:

- Use GitHub for code, documentation, metrics, and reproducible training/evaluation scripts.
- Use Hugging Face model repositories or Git LFS for trained model weights.
- Keep `model_registry.json` as the pointer between product code and the externally hosted checkpoint.
