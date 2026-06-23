"""Export the product ConvNeXt-Tiny checkpoint to ONNX for Netron inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import models


BASE_DIR = Path(__file__).parent
DEFAULT_REGISTRY_PATH = BASE_DIR / "model_registry.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "model_exports" / "bsfs_convnext_tiny_product.onnx"
DEFAULT_OPSET = 18


def load_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint_path(registry: dict) -> Path:
    checkpoint = registry["modes"]["product_3class"]["checkpoints"][0]
    candidates = [
        BASE_DIR / checkpoint["path"],
        BASE_DIR / "bsfs_convnext_tiny_final.pth",
        BASE_DIR / "checkpoints_clean_split_convnext_tiny" / "bsfs_convnext_tiny_final.pth",
    ]
    for path in candidates:
        if path.exists():
            return path

    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find checkpoint. Tried:\n{tried}")


def build_convnext_tiny(num_classes: int) -> torch.nn.Module:
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


def export_onnx(registry_path: Path, output_path: Path, opset: int) -> None:
    registry = load_registry(registry_path)
    checkpoint_path = resolve_checkpoint_path(registry)
    image_size = int(registry["preprocessing"]["image_size"])
    class_names = registry["class_names"]

    model = build_convnext_tiny(num_classes=len(class_names))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["bsfs_logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "bsfs_logits": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"Exported ONNX model: {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input: batch x 3 x {image_size} x {image_size}")
    print(f"Output logits: {len(class_names)} raw BSFS classes")
    print("Open the ONNX file in Netron to inspect the graph.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BSFS ConvNeXt-Tiny to ONNX for Netron.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = parser.parse_args()

    export_onnx(args.registry, args.output, args.opset)


if __name__ == "__main__":
    main()
