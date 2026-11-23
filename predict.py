import json
import sys

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config

classes = json.load(open("checkpoints/classes.json"))

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
model.eval()

transform = create_transform(**resolve_data_config({}, model=model))

image = Image.open(sys.argv[1]).convert("RGB")
with torch.no_grad():
    probs = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]

for p, i in zip(*probs.topk(3)):
    print(f"{classes[i]:<24}{p.item():.1%}")