"""Evaluate 3-class BSFS product grouping and Type 7 watery-risk metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


GROUP_NAMES = [
    "Type 1/2 hard",
    "Type 3/4 normal-range",
    "Type 5/6/7 loose-watery",
]
LABEL_TO_GROUP = {
    "Type 1": 0,
    "Type 2": 0,
    "Type 3": 1,
    "Type 4": 1,
    "Type 5": 2,
    "Type 6": 2,
    "Type 7": 2,
}
GROUP_TO_RAW_LABELS = {
    0: ["Type 1", "Type 2"],
    1: ["Type 3", "Type 4"],
    2: ["Type 5", "Type 6", "Type 7"],
}
TYPE7_LABEL = "Type 7"
RISK_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]


def load_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def value(row: dict, key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, ""):
        return None
    return float(raw)


def probability_for_label(row: dict, label: str) -> float | None:
    prob_key = f"prob_{label.replace(' ', '_')}"
    direct = value(row, prob_key)
    if direct is not None:
        return direct
    for rank in (1, 2, 3):
        if row.get(f"top{rank}_label") == label:
            return value(row, f"top{rank}_prob")
    return None


def group_probability(row: dict, group_idx: int) -> float | None:
    direct = value(row, f"prob_group_{group_idx}")
    if direct is not None:
        return direct

    label_key = GROUP_NAMES[group_idx].replace(" ", "_").replace("/", "_").replace("-", "_")
    direct = value(row, f"prob_{label_key}")
    if direct is not None:
        return direct

    probs = [probability_for_label(row, label) for label in GROUP_TO_RAW_LABELS[group_idx]]
    known = [prob for prob in probs if prob is not None]
    if known:
        return float(sum(known))
    return None


def predicted_group(row: dict) -> int:
    pred_label = row.get("pred_group_label")
    if pred_label in GROUP_NAMES:
        return GROUP_NAMES.index(pred_label)
    if row.get("pred_label") in LABEL_TO_GROUP:
        return LABEL_TO_GROUP[row["pred_label"]]

    probs = [group_probability(row, idx) for idx in range(len(GROUP_NAMES))]
    if all(prob is not None for prob in probs):
        return int(np.argmax(probs))
    raise ValueError(f"Cannot infer predicted group from row: {row}")


def type7_probability(row: dict) -> float:
    direct = value(row, "prob_type7")
    if direct is not None:
        return direct
    direct = value(row, "prob_Type_7")
    if direct is not None:
        return direct
    fallback = probability_for_label(row, TYPE7_LABEL)
    return float(fallback) if fallback is not None else 0.0


def risk_at_thresholds(y_true_type7: np.ndarray, y_score: np.ndarray) -> dict[str, dict[str, float]]:
    results = {}
    for threshold in RISK_THRESHOLDS:
        pred = y_score >= threshold
        tp = int(np.logical_and(pred, y_true_type7).sum())
        fp = int(np.logical_and(pred, ~y_true_type7).sum())
        fn = int(np.logical_and(~pred, y_true_type7).sum())
        tn = int(np.logical_and(~pred, ~y_true_type7).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        false_negative_rate = fn / (tp + fn) if (tp + fn) else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
        results[f"{threshold:.2f}"] = {
            "precision": precision,
            "recall": recall,
            "false_negative_rate": false_negative_rate,
            "false_positive_rate": false_positive_rate,
            "flagged_rate": float(pred.mean()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    return results


def evaluate_predictions(path: Path) -> dict:
    rows = load_rows(path)
    y_true_group = np.array([LABEL_TO_GROUP[row["true_label"]] for row in rows], dtype=np.int64)
    y_pred_group = np.array([predicted_group(row) for row in rows], dtype=np.int64)
    y_true_type7 = np.array([row["true_label"] == TYPE7_LABEL for row in rows], dtype=bool)
    y_type7_prob = np.array([type7_probability(row) for row in rows], dtype=np.float64)
    pred_group_conf = np.array([
        group_probability(row, int(pred_group)) if group_probability(row, int(pred_group)) is not None else np.nan
        for row, pred_group in zip(rows, y_pred_group)
    ])

    type7_auc = None
    type7_ap = None
    if len(np.unique(y_true_type7)) == 2:
        type7_auc = float(roc_auc_score(y_true_type7, y_type7_prob))
        type7_ap = float(average_precision_score(y_true_type7, y_type7_prob))

    cm = confusion_matrix(y_true_group, y_pred_group, labels=list(range(len(GROUP_NAMES))))
    report = classification_report(
        y_true_group,
        y_pred_group,
        labels=list(range(len(GROUP_NAMES))),
        target_names=GROUP_NAMES,
        digits=4,
        zero_division=0,
    )
    correct = y_true_group == y_pred_group
    known_conf = ~np.isnan(pred_group_conf)

    return {
        "prediction_csv": str(path),
        "groups": GROUP_NAMES,
        "total": len(rows),
        "grouped_3class_accuracy": float(accuracy_score(y_true_group, y_pred_group)),
        "grouped_3class_macro_f1": float(f1_score(y_true_group, y_pred_group, average="macro", labels=list(range(len(GROUP_NAMES))), zero_division=0)),
        "grouped_3class_weighted_f1": float(f1_score(y_true_group, y_pred_group, average="weighted", labels=list(range(len(GROUP_NAMES))), zero_division=0)),
        "grouped_3class_correct": int(correct.sum()),
        "mean_pred_group_confidence": float(np.mean(pred_group_conf[known_conf])) if known_conf.any() else None,
        "mean_pred_group_confidence_correct": float(np.mean(pred_group_conf[np.logical_and(known_conf, correct)])) if np.logical_and(known_conf, correct).any() else None,
        "type7_support": int(y_true_type7.sum()),
        "type7_probability_mean": float(np.mean(y_type7_prob)),
        "type7_probability_mean_true_type7": float(np.mean(y_type7_prob[y_true_type7])) if y_true_type7.any() else None,
        "type7_probability_mean_not_type7": float(np.mean(y_type7_prob[~y_true_type7])) if (~y_true_type7).any() else None,
        "type7_roc_auc": type7_auc,
        "type7_average_precision": type7_ap,
        "type7_risk_thresholds": risk_at_thresholds(y_true_type7, y_type7_prob),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def write_outputs(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 3-Class BSFS Product Metrics and Type 7 Risk",
        "",
        f"Prediction CSV: `{result['prediction_csv']}`",
        f"Total images: {result['total']}",
        f"3-class grouped accuracy: {100.0 * result['grouped_3class_accuracy']:.2f}%",
        f"3-class grouped macro F1: {result['grouped_3class_macro_f1']:.4f}",
        f"3-class grouped weighted F1: {result['grouped_3class_weighted_f1']:.4f}",
        f"Type 7 support: {result['type7_support']}",
        f"Type 7 ROC-AUC: {result['type7_roc_auc']:.4f}" if result["type7_roc_auc"] is not None else "Type 7 ROC-AUC: N/A",
        f"Type 7 average precision: {result['type7_average_precision']:.4f}" if result["type7_average_precision"] is not None else "Type 7 average precision: N/A",
        "",
        "## Type 7 Risk Thresholds",
        "",
        "| Threshold | Precision | Recall | FNR | FPR | Flagged | TP | FP | FN | TN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold, values in result["type7_risk_thresholds"].items():
        lines.append(
            f"| {threshold} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['false_negative_rate']:.4f} | {values['false_positive_rate']:.4f} | "
            f"{values['flagged_rate']:.4f} | {values['tp']} | {values['fp']} | {values['fn']} | {values['tn']} |"
        )
    lines += [
        "",
        "## Classification Report",
        "",
        "```text",
        result["classification_report"],
        "```",
        "",
        "## Confusion Matrix",
        "",
        "```text",
        np.array2string(np.array(result["confusion_matrix"])),
        "```",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 3-class grouped BSFS output and Type 7 risk")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate_predictions(args.predictions)
    out = args.out or args.predictions.parent / "grouped_3class_type7_risk.md"
    write_outputs(result, out)
    print(f"3-class grouped accuracy: {100.0 * result['grouped_3class_accuracy']:.2f}%")
    print(f"3-class grouped macro F1: {result['grouped_3class_macro_f1']:.4f}")
    if result["type7_roc_auc"] is not None:
        print(f"Type 7 ROC-AUC: {result['type7_roc_auc']:.4f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
