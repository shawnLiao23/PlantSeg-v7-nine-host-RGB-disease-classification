"""Small CPU checks for validation protocol and manifest alignment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18

from plantseg_cw2.evaluate import predict_model, scores


class ValidationProtocolTests(unittest.TestCase):
    def test_cpu_protocols_preserve_ids_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Metadata.csv").write_text("Name\n", encoding="utf-8")
            (root / "images" / "val").mkdir(parents=True)
            manifest = []
            for index in range(3):
                name = f"sample_{index}.png"
                pixels = np.full((240, 250, 3), 40 + 70 * index, dtype=np.uint8)
                Image.fromarray(pixels).save(root / "images" / "val" / name)
                manifest.append({"image_id": name, "image_path": f"images/val/{name}",
                                 "class_id": index % 2, "split": "val", "excluded": "0"})
            torch.manual_seed(7)
            model = resnet18(weights=None)
            model.fc = torch.nn.Linear(512, 2)
            device = torch.device("cpu")
            outputs = {}
            for protocol in ("cached_features", "feature_fp32", "full_amp"):
                metrics, predictions = predict_model(
                    model, "val", manifest, root, 2, device,
                    batch_size=2, protocol=protocol, include_margin=True)
                self.assertEqual([item["image_id"] for item in predictions],
                                 [item["image_id"] for item in manifest])
                self.assertEqual([item["true_class"] for item in predictions], [0, 1, 0])
                self.assertTrue(all(item["top2_logit_margin"] >= 0 for item in predictions))
                outputs[protocol] = [item["pred_class"] for item in predictions]
                self.assertEqual(metrics["macro_f1"],
                                 scores([0, 1, 0], outputs[protocol], 2)["macro_f1"])
            self.assertEqual(outputs["cached_features"], outputs["feature_fp32"])
            self.assertEqual(outputs["cached_features"], outputs["full_amp"])
            self.assertIsInstance(model.fc, torch.nn.Linear)


if __name__ == "__main__":
    unittest.main()
