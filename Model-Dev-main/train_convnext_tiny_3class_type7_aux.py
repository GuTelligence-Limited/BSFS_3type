"""Train ConvNeXt-Tiny with 3-class product output and Type 7 risk head."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.transforms import functional as TF

from evaluate_3class_type7_risk import GROUP_NAMES, LABEL_TO_GROUP


BASE_DIR = Path(__file__).parent
DATA_ROOT = Path(os.environ.get("BSFS_DATA_ROOT", BASE_DIR / "GuTelligence-StoMy-Clean-Split"))
OUT_DIR = Path(os.environ.get("BSFS_CHECKPOINT_DIR", BASE_DIR / "checkpoints_clean_split_convnext_tiny_3class_type7_aux"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.environ.get("BSFS_SEED", "20260601"))
IMG_SIZE = int(os.environ.get("BSFS_IMG_SIZE", "300"))
RESIZE_SIZE = int(os.environ.get("BSFS_RESIZE_SIZE", "340"))
BATCH_SIZE = int(os.environ.get("BSFS_BATCH_SIZE", "18"))
NUM_WORKERS = int(os.environ.get("BSFS_NUM_WORKERS", "4"))
HEAD_EPOCHS = int(os.environ.get("BSFS_HEAD_EPOCHS", "8"))
FULL_EPOCHS = int(os.environ.get("BSFS_FULL_EPOCHS", "22"))
PATIENCE = int(os.environ.get("BSFS_PATIENCE", "8"))
HEAD_LR = float(os.environ.get("BSFS_HEAD_LR", "5e-4"))
BACKBONE_LR = float(os.environ.get("BSFS_BACKBONE_LR", "8e-6"))
FULL_HEAD_LR = float(os.environ.get("BSFS_FULL_HEAD_LR", "4e-5"))
WEIGHT_DECAY = float(os.environ.get("BSFS_WEIGHT_DECAY", "8e-5"))
LABEL_SMOOTHING = float(os.environ.get("BSFS_LABEL_SMOOTHING", "0.02"))
TYPE7_LOSS_WEIGHT = float(os.environ.get("BSFS_TYPE7_LOSS_WEIGHT", "0.35"))

CLASS_NAMES = [f"Type {idx}" for idx in range(1, 8)]
NUM_GROUPS = len(GROUP_NAMES)
RAW_TO_GROUP = torch.tensor([LABEL_TO_GROUP[label] for label in CLASS_NAMES], dtype=torch.long)
GROUP_REPRESENTATIVE_LABEL = {
    0: "Type 2",
    1: "Type 4",
    2: "Type 6",
}


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


class ThreeClassType7ConvNeXt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[-1].in_features
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.flatten = nn.Flatten(1)
        self.group_head = nn.Linear(in_features, NUM_GROUPS)
        self.type7_head = nn.Linear(in_features, 1)
        nn.init.trunc_normal_(self.group_head.weight, std=0.02)
        nn.init.zeros_(self.group_head.bias)
        nn.init.trunc_normal_(self.type7_head.weight, std=0.02)
        nn.init.zeros_(self.type7_head.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return {
            "group_logits": self.group_head(x),
            "type7_logit": self.type7_head(x).squeeze(1),
        }


class ThreeClassType7Loss(nn.Module):
    def __init__(self, type7_weight: float, label_smoothing: float):
        super().__init__()
        self.type7_weight = type7_weight
        self.group_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.type7_bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs: dict[str, torch.Tensor], raw_labels: torch.Tensor) -> torch.Tensor:
        group_labels = RAW_TO_GROUP.to(raw_labels.device)[raw_labels]
        type7_labels = (raw_labels == CLASS_NAMES.index("Type 7")).float()
        group_loss = self.group_ce(outputs["group_logits"], group_labels)
        type7_loss = self.type7_bce(outputs["type7_logit"], type7_labels)
        return group_loss + self.type7_weight * type7_loss


def freeze_backbone(model: ThreeClassType7ConvNeXt) -> None:
    for param in model.features.parameters():
        param.requires_grad = False
    for param in list(model.group_head.parameters()) + list(model.type7_head.parameters()):
        param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.86, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=6),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.06, hue=0.015),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def tta_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return [
        transforms.Compose([transforms.Resize(RESIZE_SIZE), transforms.CenterCrop(IMG_SIZE),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize(RESIZE_SIZE), transforms.CenterCrop(IMG_SIZE),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize(RESIZE_SIZE), DeterministicSquareCrop(IMG_SIZE, "top_left"),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize(RESIZE_SIZE), DeterministicSquareCrop(IMG_SIZE, "bottom_right"),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize(RESIZE_SIZE), DeterministicSquareCrop(IMG_SIZE, "center"),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
    ]


def build_loaders():
    train_tf, eval_tf = get_transforms()
    train_ds = datasets.ImageFolder(str(DATA_ROOT / "train"), transform=train_tf)
    valid_ds = datasets.ImageFolder(str(DATA_ROOT / "valid"), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(DATA_ROOT / "test"), transform=eval_tf)

    group_labels = np.array([RAW_TO_GROUP[label].item() for _, label in train_ds.samples])
    counts = np.bincount(group_labels, minlength=NUM_GROUPS)
    weights = (1.0 / counts) ** 0.9
    sample_weights = torch.tensor([weights[group] for group in group_labels], dtype=torch.float32)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, valid_loader, test_loader, train_ds


def build_scheduler(optimizer, epochs: int):
    def lr_lambda(epoch):
        if epoch < 2:
            return float(epoch + 1) / 2.0
        progress = (epoch - 2) / max(1, epochs - 2)
        return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def prediction_outputs(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    group_probs = torch.softmax(outputs["group_logits"], dim=1)
    type7_probs = torch.sigmoid(outputs["type7_logit"])
    return group_probs, type7_probs


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    y_true_group, y_pred_group = [], []
    y_true_type7, y_type7_prob = [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        group_probs, type7_probs = prediction_outputs(outputs)
        group_labels = RAW_TO_GROUP.to(labels.device)[labels]
        pred_group = group_probs.argmax(dim=1)
        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
        correct += (pred_group == group_labels).sum().item()
        y_true_group.extend(group_labels.cpu().numpy())
        y_pred_group.extend(pred_group.cpu().numpy())
        y_true_type7.extend((labels == CLASS_NAMES.index("Type 7")).cpu().numpy())
        y_type7_prob.extend(type7_probs.cpu().numpy())
    return (
        total_loss / total,
        100.0 * correct / total,
        np.array(y_true_group),
        np.array(y_pred_group),
        np.array(y_true_type7, dtype=bool),
        np.array(y_type7_prob),
    )


def selection_score(group_acc: float, macro_f1: float, type7_auc: float) -> float:
    return 0.65 * (group_acc / 100.0) + 0.25 * macro_f1 + 0.10 * type7_auc


def metric_dict(y_true_group: np.ndarray, y_pred_group: np.ndarray, y_true_type7: np.ndarray, y_type7_prob: np.ndarray) -> dict[str, float]:
    type7_auc = float(roc_auc_score(y_true_type7, y_type7_prob)) if len(np.unique(y_true_type7)) == 2 else 0.0
    type7_ap = float(average_precision_score(y_true_type7, y_type7_prob)) if len(np.unique(y_true_type7)) == 2 else 0.0
    pred_type7 = y_type7_prob >= 0.5
    type7_recall = float(np.mean(pred_type7[y_true_type7])) if np.any(y_true_type7) else 0.0
    type7_precision = float(np.mean(y_true_type7[pred_type7])) if np.any(pred_type7) else 0.0
    return {
        "grouped_3class_accuracy": round(100.0 * accuracy_score(y_true_group, y_pred_group), 3),
        "grouped_3class_macro_f1": round(float(f1_score(y_true_group, y_pred_group, average="macro", labels=list(range(NUM_GROUPS)), zero_division=0)), 5),
        "type7_roc_auc": round(type7_auc, 5),
        "type7_average_precision": round(type7_ap, 5),
        "type7_recall_at_050": round(type7_recall, 5),
        "type7_precision_at_050": round(type7_precision, 5),
    }


def run_phase(name, model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs):
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    rows = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loss, total, correct = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            group_probs, _ = prediction_outputs(outputs)
            group_labels = RAW_TO_GROUP.to(labels.device)[labels]
            train_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            correct += (group_probs.argmax(dim=1) == group_labels).sum().item()

        val_loss, val_acc, y_true, y_pred, y_true_type7, y_type7_prob = evaluate(model, valid_loader, criterion)
        macro_f1 = f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_GROUPS)), zero_division=0)
        type7_auc = roc_auc_score(y_true_type7, y_type7_prob) if len(np.unique(y_true_type7)) == 2 else 0.0
        score = selection_score(val_acc, macro_f1, type7_auc)
        scheduler.step()

        row = {
            "epoch": epoch,
            "phase": name,
            "train_loss": round(train_loss / total, 5),
            "train_acc": round(100.0 * correct / total, 3),
            "val_loss": round(val_loss, 5),
            "val_group_acc": round(val_acc, 3),
            "val_group_macro_f1": round(float(macro_f1), 5),
            "val_type7_auc": round(float(type7_auc), 5),
            "selection_score": round(float(score), 5),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 9),
            "elapsed_sec": round(time.time() - start, 1),
        }
        rows.append(row)
        print(
            f"{name} {epoch:02d}/{epochs} | train {row['train_acc']:.1f}% "
            f"| val group {val_acc:.1f}% f1 {macro_f1:.3f} t7auc {type7_auc:.3f} "
            f"sel {score:.4f} | {row['elapsed_sec']:.0f}s"
        )

        if score > best_score + 5e-4:
            best_score = float(score)
            best_epoch = epoch
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, OUT_DIR / f"best_{name}.pth")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping {name} at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return model, rows, best_epoch, best_score


@torch.no_grad()
def predict_tta(model, split: str):
    dataset = datasets.ImageFolder(str(DATA_ROOT / split))
    paths = [Path(path) for path, _ in dataset.samples]
    labels = np.array([label for _, label in dataset.samples])
    sum_group_probs = None
    sum_type7_probs = None
    for idx, tf in enumerate(tta_transforms(), start=1):
        dataset.transform = tf
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        group_probs_list = []
        type7_probs_list = []
        for inputs, _ in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            group_probs, type7_probs = prediction_outputs(outputs)
            group_probs_list.append(group_probs.cpu().numpy())
            type7_probs_list.append(type7_probs.cpu().numpy())
        view_group_probs = np.vstack(group_probs_list)
        view_type7_probs = np.concatenate(type7_probs_list)
        sum_group_probs = view_group_probs if sum_group_probs is None else sum_group_probs + view_group_probs
        sum_type7_probs = view_type7_probs if sum_type7_probs is None else sum_type7_probs + view_type7_probs
        print(f"TTA view {idx}/5 done")
    return sum_group_probs / 5.0, sum_type7_probs / 5.0, labels, paths


def write_predictions(
    path: Path,
    paths: list[Path],
    y_true_raw: np.ndarray,
    y_pred_group: np.ndarray,
    group_probs: np.ndarray,
    type7_probs: np.ndarray,
) -> None:
    rows = []
    for idx, image_path in enumerate(paths):
        pred_group = int(y_pred_group[idx])
        pred_label = GROUP_REPRESENTATIVE_LABEL[pred_group]
        top = np.argsort(group_probs[idx])[::-1]
        row = {
            "image_path": str(image_path),
            "rel_path": image_path.relative_to(DATA_ROOT).as_posix(),
            "true_label": CLASS_NAMES[int(y_true_raw[idx])],
            "pred_label": pred_label,
            "pred_group_label": GROUP_NAMES[pred_group],
            "correct": bool(LABEL_TO_GROUP[CLASS_NAMES[int(y_true_raw[idx])]] == pred_group),
            "confidence": float(group_probs[idx, pred_group]),
            "top1_label": pred_label,
            "top1_prob": float(group_probs[idx, pred_group]),
            "top2_label": GROUP_REPRESENTATIVE_LABEL[int(top[1])],
            "top2_prob": float(group_probs[idx, top[1]]),
            "top3_label": GROUP_REPRESENTATIVE_LABEL[int(top[2])],
            "top3_prob": float(group_probs[idx, top[2]]),
            "ordinal_abs_error": abs(int(y_true_raw[idx]) - CLASS_NAMES.index(pred_label)),
            "prob_group_0": float(group_probs[idx, 0]),
            "prob_group_1": float(group_probs[idx, 1]),
            "prob_group_2": float(group_probs[idx, 2]),
            "prob_type7": float(type7_probs[idx]),
        }
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Data: {DATA_ROOT}")
    print(f"Output: {OUT_DIR}")
    print(f"Image size: {IMG_SIZE} | Resize: {RESIZE_SIZE} | Batch: {BATCH_SIZE}")
    print(f"Type 7 loss weight: {TYPE7_LOSS_WEIGHT}")

    train_loader, valid_loader, test_loader, train_ds = build_loaders()
    print(f"Train: {len(train_ds)} | Valid: {len(valid_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = ThreeClassType7ConvNeXt().to(DEVICE)
    criterion = ThreeClassType7Loss(
        type7_weight=TYPE7_LOSS_WEIGHT,
        label_smoothing=LABEL_SMOOTHING,
    ).to(DEVICE)
    history = {"head": [], "full": []}

    freeze_backbone(model)
    optimizer_head = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )
    model, rows, best_head_epoch, best_head_score = run_phase(
        "head", model, train_loader, valid_loader, criterion,
        optimizer_head, build_scheduler(optimizer_head, HEAD_EPOCHS), HEAD_EPOCHS,
    )
    history["head"] = rows

    unfreeze_all(model)
    optimizer_full = optim.AdamW([
        {"params": model.features.parameters(), "lr": BACKBONE_LR},
        {"params": list(model.group_head.parameters()) + list(model.type7_head.parameters()), "lr": FULL_HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    model, rows, best_full_epoch, best_full_score = run_phase(
        "full", model, train_loader, valid_loader, criterion,
        optimizer_full, build_scheduler(optimizer_full, FULL_EPOCHS), FULL_EPOCHS,
    )
    history["full"] = rows

    test_loss, _, y_true_group, y_pred_group, y_true_type7, y_type7_prob = evaluate(model, test_loader, criterion)
    single_metrics = metric_dict(y_true_group, y_pred_group, y_true_type7, y_type7_prob)
    group_probs, type7_probs, y_raw_tta, paths = predict_tta(model, "test")
    y_pred_group_tta = group_probs.argmax(axis=1)
    y_true_group_tta = np.array([RAW_TO_GROUP[int(label)].item() for label in y_raw_tta])
    y_true_type7_tta = np.array([CLASS_NAMES[int(label)] == "Type 7" for label in y_raw_tta])
    tta_metrics = metric_dict(y_true_group_tta, y_pred_group_tta, y_true_type7_tta, type7_probs)

    report = classification_report(
        y_true_group_tta,
        y_pred_group_tta,
        target_names=GROUP_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true_group_tta, y_pred_group_tta, labels=list(range(NUM_GROUPS)))
    write_predictions(OUT_DIR / "all_predictions_tta.csv", paths, y_raw_tta, y_pred_group_tta, group_probs, type7_probs)

    result = {
        "experiment": "convnext_tiny_3class_type7_aux",
        "data_root": str(DATA_ROOT),
        "img_size": IMG_SIZE,
        "resize_size": RESIZE_SIZE,
        "batch_size": BATCH_SIZE,
        "type7_loss_weight": TYPE7_LOSS_WEIGHT,
        "label_smoothing": LABEL_SMOOTHING,
        "best_head_epoch": best_head_epoch,
        "best_head_score": best_head_score,
        "best_full_epoch": best_full_epoch,
        "best_full_score": best_full_score,
        "history": history,
        "test_single": {"loss": round(float(test_loss), 5), **single_metrics},
        "test_tta": tta_metrics,
        "classification_report_tta": report,
        "confusion_matrix_tta": cm.tolist(),
    }
    (OUT_DIR / "training_history.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), OUT_DIR / "bsfs_convnext_tiny_3class_type7_aux_final.pth")

    summary = [
        "# ConvNeXt-Tiny 3-Class + Type 7 Auxiliary Experiment",
        "",
        f"Data root: `{DATA_ROOT}`",
        f"Image size: `{IMG_SIZE}`; resize: `{RESIZE_SIZE}`",
        f"Type 7 loss weight: `{TYPE7_LOSS_WEIGHT}`",
        f"Best head epoch: `{best_head_epoch}`; best head selection score: `{best_head_score:.5f}`",
        f"Best full epoch: `{best_full_epoch}`; best full selection score: `{best_full_score:.5f}`",
        "",
        "## Test Metrics",
        "",
        f"Single-view 3-class accuracy: {single_metrics['grouped_3class_accuracy']:.2f}%",
        f"Single-view 3-class macro F1: {single_metrics['grouped_3class_macro_f1']:.4f}",
        f"Single-view Type 7 ROC-AUC: {single_metrics['type7_roc_auc']:.4f}",
        f"TTA 3-class accuracy: {tta_metrics['grouped_3class_accuracy']:.2f}%",
        f"TTA 3-class macro F1: {tta_metrics['grouped_3class_macro_f1']:.4f}",
        f"TTA Type 7 ROC-AUC: {tta_metrics['type7_roc_auc']:.4f}",
        f"TTA Type 7 AP: {tta_metrics['type7_average_precision']:.4f}",
        "",
        "## TTA Classification Report",
        "",
        "```text",
        report,
        "```",
        "",
        "## TTA Confusion Matrix",
        "",
        "```text",
        np.array2string(cm),
        "```",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"TTA 3-class accuracy: {tta_metrics['grouped_3class_accuracy']:.2f}%")
    print(f"TTA Type 7 ROC-AUC: {tta_metrics['type7_roc_auc']:.4f}")
    print(report)


if __name__ == "__main__":
    main()
