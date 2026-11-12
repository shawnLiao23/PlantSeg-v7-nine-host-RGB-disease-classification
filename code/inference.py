"""Load the selected checkpoint and classify one RGB image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from torchvision.models import resnet18

from dataset import make_transforms, result_root
from train import device


def predict(image_path: str | Path, checkpoint_path: str | Path | None = None,
            top_k: int = 5, output_dir: str | Path | None = None) -> list[dict]:
    output = result_root(output_dir)
    if checkpoint_path is None:
        plan = json.loads((output / "frozen_plan.json").read_text(encoding="utf-8"))
        name = plan["selected_model"]
        if name == "majority_class":
            raise ValueError("The majority-class predictor has no image checkpoint")
        checkpoint_path = output / "checkpoints" / f"{name}_best.pth"
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    classes = payload["class_mapping"]["classes"]
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(classes))
    model.load_state_dict(payload["model_state_dict"])
    target_device = device()
    model.to(target_device).eval()
    with Image.open(Path(image_path)) as image:
        tensor = make_transforms(False)(image.convert("RGB")).unsqueeze(0).to(target_device)
    feature_model = torch.nn.Sequential(*list(model.children())[:-1]).eval()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", enabled=target_device.type == "cuda"):
            features = feature_model(tensor)
        logits = model.fc(features.flatten(1).float())
        probabilities = torch.softmax(logits.float(), dim=1)[0]
    values, indices = probabilities.topk(min(top_k, len(classes)))
    return [
        {"class_id": int(index), "class_name": classes[int(index)]["class_name"],
         "softmax_score": float(value)}
        for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    print(json.dumps({"image": str(args.image), "predictions": predict(
        args.image, args.checkpoint, args.top_k)}, indent=2))
    print("Softmax scores are not calibrated probabilities.")


if __name__ == "__main__":
    main()
