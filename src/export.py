import json
from pathlib import Path

import timm
import torch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "best.pt"
CLASSES_FILE = ROOT / "checkpoints" / "classes.json"
OUTPUT = ROOT / "checkpoints" / "pet-breed-classifier.onnx"


def main():
    classes = json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=len(classes),
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        OUTPUT,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["breed_logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "breed_logits": {0: "batch"},
        },
    )
    print(f"Exported {OUTPUT}")


if __name__ == "__main__":
    main()
