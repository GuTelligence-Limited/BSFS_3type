from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import functional as TF


BASE_DIR = Path(__file__).parent
REGISTRY_PATH = BASE_DIR / "model_registry.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    def __call__(self, img: Image.Image) -> Image.Image:
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


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def tta_transforms(config: dict) -> list[transforms.Compose]:
    image_size = int(config["image_size"])
    resize_size = int(config["resize_size"])
    mean = config["mean"]
    std = config["std"]
    return [
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Resize(resize_size),
            DeterministicSquareCrop(image_size, "top_left"),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Resize(resize_size),
            DeterministicSquareCrop(image_size, "bottom_right"),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Resize(resize_size),
            DeterministicSquareCrop(image_size, "center"),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    ]


def build_convnext_tiny(num_classes: int) -> torch.nn.Module:
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


@lru_cache(maxsize=1)
def load_model_and_config() -> tuple[torch.nn.Module, dict, dict]:
    registry = load_registry()
    mode_config = registry["modes"]["product_3class"]
    checkpoint = mode_config["checkpoints"][0]
    checkpoint_path = BASE_DIR / checkpoint["path"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. Upload the .pth file with Git LFS."
        )

    model = build_convnext_tiny(num_classes=len(registry["class_names"]))
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model, registry, mode_config


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
            "reason": "Use continuous Type 7 probability; no fixed alert threshold is approved yet.",
        },
        "raw_top1_label": class_names[pred_idx],
        "raw_top1_confidence": float(probs[pred_idx]),
        "raw_probabilities": {
            class_names[idx]: float(probs[idx])
            for idx in range(len(class_names))
        },
    }


@torch.no_grad()
def predict(image: Image.Image) -> tuple[str, float, float, dict, dict]:
    if image is None:
        raise gr.Error("Please upload an image.")

    model, registry, _mode_config = load_model_and_config()
    class_names = registry["class_names"]
    rgb_image = image.convert("RGB")

    view_probs = []
    for transform in tta_transforms(registry["preprocessing"]):
        inputs = transform(rgb_image).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(model(inputs), dim=1).cpu().numpy()[0]
        view_probs.append(probs)

    probs = np.mean(np.vstack(view_probs), axis=0)
    order = np.argsort(probs)[::-1]
    pred_idx = int(order[0])
    schema = product_schema(probs, pred_idx, class_names)
    top3 = {
        class_names[int(idx)]: float(probs[idx])
        for idx in order[:3]
    }

    return (
        schema["primary_group"],
        round(schema["primary_group_confidence"], 4),
        round(schema["type7_probability"], 4),
        schema["group_probabilities"],
        {
            "product_schema": schema,
            "top3": top3,
            "device": DEVICE,
            "registry_version": registry["version"],
        },
    )


with gr.Blocks(title="BSFS 3-Class + Type 7 Risk MVP") as demo:
    gr.Markdown(
        "# BSFS 3-Class + Type 7 Risk MVP\n"
        "Upload an image to get the product 3-class group and continuous Type 7 risk probability. "
        "This MVP is for engineering validation only, not medical diagnosis."
    )
    with gr.Row():
        image_input = gr.Image(type="pil", label="Input image")
        with gr.Column():
            primary_group = gr.Textbox(label="Primary product group")
            primary_confidence = gr.Number(label="Primary group confidence", precision=4)
            type7_probability = gr.Number(label="Type 7 probability", precision=4)
            group_probabilities = gr.JSON(label="3-class group probabilities")
    full_json = gr.JSON(label="Full product output")
    predict_button = gr.Button("Run inference", variant="primary")
    predict_button.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            primary_group,
            primary_confidence,
            type7_probability,
            group_probabilities,
            full_json,
        ],
    )


if __name__ == "__main__":
    demo.launch()
