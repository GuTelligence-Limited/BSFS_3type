"""Calibrate Type 7 watery-risk thresholds on validation predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF

from evaluate_3class_type7_risk import evaluate_predictions, write_outputs


BASE_DIR = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = [f"Type {idx}" for idx in range(1, 8)]
GROUP_NAMES_3 = [
    "Type 1/2 hard",
    "Type 3/4 normal-range",
    "Type 5/6/7 loose-watery",
]
RAW_TO_GROUP_3 = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 2,
}
RISK_THRESHOLDS = [round(x, 2) for x in np.arange(0.02, 0.52, 0.02)]


class DeterministicSquareCrop:
    def __init__(self, size: int, position: str):
        self.size = size
        self.position = position

    def __call__(self, img):
        width, height = img.size
        top = {
            "top_left": 0,
            "center": max((height - self.size) // 2, 0),
            "bottom_right": max(height - self.size, 0),
        }[self.position]
        left = {
            "top_left": 0,
            "center": max((width - self.size) // 2, 0),
            "bottom_right": max(width - self.size, 0),
        }[self.position]
        return TF.crop(img, top, left, self.size, self.size)


def tta_transforms(image_size: int, resize_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return [
        transforms.Compose([transforms.Resize(resize_size), transforms.CenterCrop(image_size),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize(resize_size), transforms.CenterCrop(image_size),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize(resize_size), DeterministicSquareCrop(image_size, "top_left"),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize(resize_size), DeterministicSquareCrop(image_size, "bottom_right"),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize(resize_size), DeterministicSquareCrop(image_size, "center"),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
    ]


def load_convnext_tiny(checkpoint: Path) -> torch.nn.Module:
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def export_predictions(
    model: torch.nn.Module,
    data_root: Path,
    split: str,
    out_csv: Path,
    image_size: int,
    resize_size: int,
    batch_size: int,
) -> None:
    dataset = datasets.ImageFolder(str(data_root / split))
    paths = [Path(path) for path, _ in dataset.samples]
    labels = np.array([label for _, label in dataset.samples])
    sum_probs = None
    for tf in tta_transforms(image_size, resize_size):
        dataset.transform = tf
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        probs = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(DEVICE)
                logits = model(inputs)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        view_probs = np.vstack(probs)
        sum_probs = view_probs if sum_probs is None else sum_probs + view_probs
    y_prob = sum_probs / 5.0
    y_pred = y_prob.argmax(axis=1)

    rows = []
    for idx, image_path in enumerate(paths):
        true_idx = int(labels[idx])
        pred_idx = int(y_pred[idx])
        group_probs = np.array([
            y_prob[idx, [0, 1]].sum(),
            y_prob[idx, [2, 3]].sum(),
            y_prob[idx, [4, 5, 6]].sum(),
        ])
        pred_group = int(RAW_TO_GROUP_3[pred_idx])
        top = np.argsort(y_prob[idx])[::-1][:3]
        row = {
            "image_path": str(image_path),
            "rel_path": image_path.relative_to(data_root).as_posix(),
            "true_label": CLASS_NAMES[true_idx],
            "pred_label": CLASS_NAMES[pred_idx],
            "pred_group_label": GROUP_NAMES_3[pred_group],
            "correct": bool(true_idx == pred_idx),
            "confidence": float(y_prob[idx, pred_idx]),
            "top1_label": CLASS_NAMES[int(top[0])],
            "top1_prob": float(y_prob[idx, top[0]]),
            "top2_label": CLASS_NAMES[int(top[1])],
            "top2_prob": float(y_prob[idx, top[1]]),
            "top3_label": CLASS_NAMES[int(top[2])],
            "top3_prob": float(y_prob[idx, top[2]]),
            "ordinal_abs_error": int(abs(true_idx - pred_idx)),
            "prob_group_0": float(group_probs[0]),
            "prob_group_1": float(group_probs[1]),
            "prob_group_2": float(group_probs[2]),
            "prob_type7": float(y_prob[idx, 6]),
        }
        for class_idx, class_name in enumerate(CLASS_NAMES):
            row[f"prob_{class_name.replace(' ', '_')}"] = float(y_prob[idx, class_idx])
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    pred = y_score >= threshold
    tp = int(np.logical_and(pred, y_true).sum())
    fp = int(np.logical_and(pred, ~y_true).sum())
    fn = int(np.logical_and(~pred, y_true).sum())
    tn = int(np.logical_and(~pred, ~y_true).sum())
    return {
        "threshold": threshold,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "false_negative_rate": fn / (tp + fn) if (tp + fn) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "flagged_rate": float(pred.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def load_type7_arrays(prediction_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    with prediction_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y_true = np.array([row["true_label"] == "Type 7" for row in rows], dtype=bool)
    y_score = np.array([float(row["prob_type7"]) for row in rows], dtype=np.float64)
    return y_true, y_score


def select_threshold(valid_csv: Path, min_recall: float) -> tuple[dict, list[dict]]:
    y_true, y_score = load_type7_arrays(valid_csv)
    rows = [threshold_metrics(y_true, y_score, threshold) for threshold in RISK_THRESHOLDS]
    feasible = [row for row in rows if row["recall"] >= min_recall]
    if feasible:
        selected = max(feasible, key=lambda row: (row["precision"], -row["false_positive_rate"], row["threshold"]))
    else:
        selected = max(rows, key=lambda row: (row["recall"], row["precision"], -row["false_positive_rate"]))
    return selected, rows


def write_calibration_report(out_dir: Path, valid_csv: Path, test_csv: Path, selected: dict, valid_rows: list[dict]) -> None:
    valid_true, valid_score = load_type7_arrays(valid_csv)
    test_true, test_score = load_type7_arrays(test_csv)
    test_selected = threshold_metrics(test_true, test_score, selected["threshold"])
    valid_auc = float(roc_auc_score(valid_true, valid_score)) if len(np.unique(valid_true)) == 2 else None
    test_auc = float(roc_auc_score(test_true, test_score)) if len(np.unique(test_true)) == 2 else None
    valid_ap = float(average_precision_score(valid_true, valid_score)) if len(np.unique(valid_true)) == 2 else None
    test_ap = float(average_precision_score(test_true, test_score)) if len(np.unique(test_true)) == 2 else None
    result = {
        "valid_csv": str(valid_csv),
        "test_csv": str(test_csv),
        "selection_policy": "max precision subject to validation recall target",
        "selected_threshold": selected["threshold"],
        "selected_valid_metrics": selected,
        "selected_test_metrics": test_selected,
        "valid_roc_auc": valid_auc,
        "test_roc_auc": test_auc,
        "valid_average_precision": valid_ap,
        "test_average_precision": test_ap,
        "valid_threshold_table": valid_rows,
    }
    (out_dir / "type7_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Type 7 Risk Calibration",
        "",
        f"Validation CSV: `{valid_csv}`",
        f"Test CSV: `{test_csv}`",
        "Selection policy: max precision subject to validation recall target.",
        f"Selected threshold: `{selected['threshold']:.2f}`",
        "",
        "## ROC / AP",
        "",
        f"Validation ROC-AUC: {valid_auc:.4f}" if valid_auc is not None else "Validation ROC-AUC: N/A",
        f"Validation AP: {valid_ap:.4f}" if valid_ap is not None else "Validation AP: N/A",
        f"Test ROC-AUC: {test_auc:.4f}" if test_auc is not None else "Test ROC-AUC: N/A",
        f"Test AP: {test_ap:.4f}" if test_ap is not None else "Test AP: N/A",
        "",
        "## Selected Threshold Performance",
        "",
        "| Split | Threshold | Precision | Recall | FNR | FPR | Flagged | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        format_threshold_row("valid", selected),
        format_threshold_row("test", test_selected),
        "",
        "## Validation Threshold Table",
        "",
        "| Threshold | Precision | Recall | FNR | FPR | Flagged | TP | FP | FN | TN |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(format_threshold_row("", row, include_split=False) for row in valid_rows)
    (out_dir / "type7_calibration.md").write_text("\n".join(lines), encoding="utf-8")


def format_threshold_row(split: str, row: dict, include_split: bool = True) -> str:
    cells = [
        f"{row['threshold']:.2f}",
        f"{row['precision']:.4f}",
        f"{row['recall']:.4f}",
        f"{row['false_negative_rate']:.4f}",
        f"{row['false_positive_rate']:.4f}",
        f"{row['flagged_rate']:.4f}",
        str(row["tp"]),
        str(row["fp"]),
        str(row["fn"]),
        str(row["tn"]),
    ]
    if include_split:
        cells.insert(0, split)
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Type 7 risk threshold for the best ConvNeXt-Tiny model")
    parser.add_argument("--data-root", type=Path, default=BASE_DIR / "GuTelligence-StoMy-Clean-Split")
    parser.add_argument("--checkpoint", type=Path, default=BASE_DIR / "checkpoints_clean_split_convnext_tiny" / "bsfs_convnext_tiny_final.pth")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "type7_risk_calibration")
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--resize-size", type=int, default=340)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--min-recall", type=float, default=0.50)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = load_convnext_tiny(args.checkpoint)
    valid_csv = args.out_dir / "valid_predictions_full_probs.csv"
    test_csv = args.out_dir / "test_predictions_full_probs.csv"
    export_predictions(model, args.data_root, "valid", valid_csv, args.image_size, args.resize_size, args.batch_size)
    export_predictions(model, args.data_root, "test", test_csv, args.image_size, args.resize_size, args.batch_size)

    valid_eval = evaluate_predictions(valid_csv)
    test_eval = evaluate_predictions(test_csv)
    write_outputs(valid_eval, args.out_dir / "valid_grouped_3class_type7_risk.md")
    write_outputs(test_eval, args.out_dir / "test_grouped_3class_type7_risk.md")

    selected, valid_rows = select_threshold(valid_csv, args.min_recall)
    write_calibration_report(args.out_dir, valid_csv, test_csv, selected, valid_rows)
    print(f"Selected Type 7 threshold: {selected['threshold']:.2f}")
    print(f"Validation recall: {selected['recall']:.4f} | precision: {selected['precision']:.4f}")
    print(f"Saved: {args.out_dir / 'type7_calibration.md'}")


if __name__ == "__main__":
    main()
