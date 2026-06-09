"""Single-image BSFS inference wrapper with selective abstention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import functional as TF

from model import BSFSClassifier


BASE_DIR = Path(__file__).parent
REGISTRY_PATH = BASE_DIR / "model_registry.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TYPE34_INDICES = {2, 3}
GROUP_NAMES = [
    "Type 1/2 hard",
    "Type 3/4 normal-range",
    "Type 5/6 soft-loose",
    "Type 7 watery",
]
GROUP_INDICES = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6],
}
RAW_INDEX_TO_GROUP_INDEX = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 3,
}
PRODUCT_GROUP_NAMES = [
    "Type 1/2 hard",
    "Type 3/4 normal-range",
    "Type 5/6/7 loose-watery",
]
PRODUCT_GROUP_INDICES = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5, 6],
}
RAW_INDEX_TO_PRODUCT_GROUP_INDEX = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 2,
}
PRODUCT_GROUP_SCORE_ANCHORS = {
    "Type 1/2 hard": 1.5,
    "Type 3/4 normal-range": 3.5,
    "Type 5/6/7 loose-watery": 6.0,
}


class DeterministicSquareCrop:
    def __init__(self, size: int, position: str):
        self.size = size
        self.position = position

    def __call__(self, img):
        width, height = img.size
        top_map = {
            "top_left": 0,
            "center": max((height - self.size) // 2, 0),
            "bottom_right": max(height - self.size, 0),
        }
        left_map = {
            "top_left": 0,
            "center": max((width - self.size) // 2, 0),
            "bottom_right": max(width - self.size, 0),
        }
        return TF.crop(img, top_map[self.position], left_map[self.position], self.size, self.size)


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tta_transforms(config: dict):
    image_size = int(config["image_size"])
    resize_size = int(config["resize_size"])
    mean = config["mean"]
    std = config["std"]
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


def load_model(checkpoint_path: Path, num_classes: int, model_type: str = "efficientnet_b4") -> torch.nn.Module:
    if model_type == "efficientnet_b4":
        model = BSFSClassifier(num_classes=num_classes, dropout_rate=0.0, use_gem=True)
    elif model_type == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def clinical_label_and_confidence(pred_idx: int, probs: np.ndarray, class_names: list[str]) -> tuple[str, float, dict]:
    if pred_idx in TYPE34_INDICES:
        return (
            "Type 3/4 normal-range",
            float(probs[2] + probs[3]),
            {
                "rule": "type3_type4_relaxed",
                "raw_labels_grouped": [class_names[2], class_names[3]],
                "type3_probability": float(probs[2]),
                "type4_probability": float(probs[3]),
            },
        )

    return (
        class_names[pred_idx],
        float(probs[pred_idx]),
        {
            "rule": "strict_label",
            "raw_labels_grouped": [class_names[pred_idx]],
        },
    )


def grouped_label_and_confidence(probs: np.ndarray, pred_idx: int) -> tuple[str, float, dict]:
    group_probs = {
        GROUP_NAMES[group_idx]: float(probs[indices].sum())
        for group_idx, indices in GROUP_INDICES.items()
    }
    group_idx = RAW_INDEX_TO_GROUP_INDEX[pred_idx]
    group_name = GROUP_NAMES[group_idx]
    group_sum_argmax = max(group_probs, key=group_probs.get)
    return (
        group_name,
        group_probs[group_name],
        {
            "rule": "raw_top1_label_mapped_to_bsfs_4class",
            "groups": {
                GROUP_NAMES[group_idx]: indices
                for group_idx, indices in GROUP_INDICES.items()
            },
            "group_probabilities": group_probs,
            "group_sum_argmax_label": group_sum_argmax,
            "group_sum_argmax_confidence": group_probs[group_sum_argmax],
        },
    )


def product_schema(probs: np.ndarray, pred_idx: int, class_names: list[str]) -> dict:
    group_probabilities = {
        PRODUCT_GROUP_NAMES[group_idx]: float(probs[indices].sum())
        for group_idx, indices in PRODUCT_GROUP_INDICES.items()
    }
    raw_top1_group_idx = RAW_INDEX_TO_PRODUCT_GROUP_INDEX[pred_idx]
    raw_top1_group = PRODUCT_GROUP_NAMES[raw_top1_group_idx]
    probability_argmax_group = max(group_probabilities, key=group_probabilities.get)
    bsfs_continuous_score = float(
        sum(probs[idx] * (idx + 1) for idx in range(len(class_names)))
    )
    product_group_score = float(
        sum(
            group_probabilities[group_name] * PRODUCT_GROUP_SCORE_ANCHORS[group_name]
            for group_name in PRODUCT_GROUP_NAMES
        )
    )
    return {
        "schema_version": "bsfs_product_3class_v1",
        "primary_group": raw_top1_group,
        "primary_group_confidence": group_probabilities[raw_top1_group],
        "primary_group_rule": "raw_top1_label_mapped_to_3class_product_group",
        "group_probabilities": group_probabilities,
        "probability_argmax_group": probability_argmax_group,
        "probability_argmax_group_confidence": group_probabilities[probability_argmax_group],
        "bsfs_continuous_score": bsfs_continuous_score,
        "product_group_score": product_group_score,
        "type7_probability": float(probs[6]),
        "type7_risk": {
            "score": float(probs[6]),
            "threshold_status": "not_calibrated_for_fixed_flag",
            "reason": "validation-selected Type 7 thresholds did not generalize to clean test",
        },
        "raw_top1_label": class_names[pred_idx],
        "raw_top1_confidence": float(probs[pred_idx]),
        "raw_probabilities": {
            class_names[idx]: float(probs[idx])
            for idx in range(len(class_names))
        },
    }


@torch.no_grad()
def predict_image(image_path: Path, mode: str, registry: dict) -> dict:
    if mode not in registry["modes"]:
        raise ValueError(f"Unknown mode '{mode}'. Available: {list(registry['modes'])}")

    mode_config = registry["modes"][mode]
    class_names = registry["class_names"]
    image = Image.open(image_path).convert("RGB")
    transforms_list = tta_transforms(registry["preprocessing"])

    weighted_probs = None
    weight_sum = 0.0
    for checkpoint in mode_config["checkpoints"]:
        weight = float(checkpoint["weight"])
        model_type = checkpoint.get("model_type", "efficientnet_b4")
        model = load_model(BASE_DIR / checkpoint["path"], len(class_names), model_type)
        view_probs = []
        for tf in transforms_list:
            inputs = tf(image).unsqueeze(0).to(DEVICE)
            probs = torch.softmax(model(inputs), dim=1).cpu().numpy()[0]
            view_probs.append(probs)
        probs = np.mean(np.vstack(view_probs), axis=0)
        weighted_probs = probs * weight if weighted_probs is None else weighted_probs + probs * weight
        weight_sum += weight

    probs = weighted_probs / max(weight_sum, 1e-8)
    order = np.argsort(probs)[::-1]
    pred_idx = int(order[0])
    top1 = float(probs[order[0]])
    top2 = float(probs[order[1]])
    margin = top1 - top2
    accepted = (
        top1 >= float(mode_config["confidence_threshold"])
        and margin >= float(mode_config["margin_threshold"])
    )
    abstain_reasons = []
    if top1 < float(mode_config["confidence_threshold"]):
        abstain_reasons.append("low_confidence")
    if margin < float(mode_config["margin_threshold"]):
        abstain_reasons.append("low_margin")
    warnings = []
    if pred_idx in (2, 3) or float(probs[2] + probs[3]) >= 0.50:
        warnings.append("type3_type4_boundary_known_unreliable")
    clinical_label, clinical_confidence, clinical_info = clinical_label_and_confidence(
        pred_idx, probs, class_names
    )
    group_label, group_confidence, group_info = grouped_label_and_confidence(probs, pred_idx)
    product_output = product_schema(probs, pred_idx, class_names)

    return {
        "image_path": str(image_path),
        "mode": mode,
        "accepted": bool(accepted),
        "product_schema": product_output,
        "primary_group": product_output["primary_group"] if accepted else None,
        "primary_group_confidence": product_output["primary_group_confidence"],
        "bsfs_continuous_score": product_output["bsfs_continuous_score"],
        "product_group_score": product_output["product_group_score"],
        "type7_probability": product_output["type7_probability"],
        "pred_label": class_names[pred_idx] if accepted else None,
        "clinical_pred_label": clinical_label if accepted else None,
        "group_pred_label": group_label if accepted else None,
        "raw_pred_label": class_names[pred_idx],
        "confidence": top1,
        "clinical_confidence": clinical_confidence,
        "group_confidence": group_confidence,
        "margin_top1_top2": margin,
        "abstain_reasons": abstain_reasons,
        "warnings": warnings,
        "top3": [
            {"label": class_names[int(idx)], "probability": float(probs[idx])}
            for idx in order[:3]
        ],
        "probabilities": {
            class_names[idx]: float(probs[idx])
            for idx in range(len(class_names))
        },
        "clinical_interpretation": clinical_info,
        "group_interpretation": group_info,
        "registry_version": registry["version"],
        "source_result": mode_config.get("source_result", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BSFS inference on one image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mode", default="product_3class")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    result = predict_image(args.image, args.mode, registry)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
