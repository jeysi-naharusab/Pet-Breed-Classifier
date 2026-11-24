import json
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "pet-breed-classifier.onnx"
CLASSES_FILE = ROOT / "checkpoints" / "classes.json"
METADATA_FILE = ROOT / "breed_metadata.json"
PWA_DIR = ROOT / "pwa"

classes = json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
breed_metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

session = ort.InferenceSession(
    str(CHECKPOINT),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name


def preprocess(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    scale = 256 / min(width, height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    rgb_image = rgb_image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

    left = (resized_width - 224) // 2
    top = (resized_height - 224) // 2
    rgb_image = rgb_image.crop((left, top, left + 224, top + 224))

    pixels = np.asarray(rgb_image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    pixels = (pixels - mean) / std
    return pixels.transpose(2, 0, 1)[np.newaxis, ...]

app = FastAPI(title="Pet Breed Classifier API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def classify_color(image: Image.Image) -> tuple[str, float]:
    """Estimate a coarse dominant coat color from the center of the photo."""
    width, height = image.size
    center = image.crop((width * 0.15, height * 0.15, width * 0.85, height * 0.85))
    pixels = center.convert("RGB").resize((32, 32)).getdata()
    counts: Counter[str] = Counter()

    for red, green, blue in pixels:
        maximum = max(red, green, blue)
        minimum = min(red, green, blue)
        luminance = (red + green + blue) / 3
        saturation = maximum - minimum

        if luminance < 45:
            name = "Black"
        elif luminance > 220 and saturation < 40:
            name = "White"
        elif saturation < 25:
            name = "Gray"
        elif red > green * 1.25 and red > blue * 1.35:
            name = "Red"
        elif red > blue * 1.15 and green > blue * 1.15:
            if luminance > 155:
                name = "Cream"
            elif green > red * 0.65:
                name = "Golden"
            else:
                name = "Brown"
        elif blue > red * 1.15:
            name = "Blue"
        elif red > blue and green > blue:
            name = "Tan"
        else:
            name = "Mixed"

        counts[name] += 1

    color, pixel_count = counts.most_common(1)[0]
    confidence = pixel_count / sum(counts.values())
    return color, round(confidence, 3)


def predict(image: Image.Image) -> dict:
    rgb_image = image.convert("RGB")
    logits = session.run(None, {input_name: preprocess(rgb_image)})[0][0]
    logits -= np.max(logits)
    probabilities = np.exp(logits) / np.sum(np.exp(logits))

    top_index = int(np.argmax(probabilities))
    top_probability = float(probabilities[top_index])
    breed = classes[top_index]
    metadata = breed_metadata.get(breed, {"species": "Unknown", "size": "Unknown"})
    color, color_confidence = classify_color(rgb_image)

    topk_indices = np.argsort(probabilities)[-min(3, len(classes)):][::-1]
    alternatives = [
        {"breed": classes[index], "confidence": round(float(probabilities[index]), 3)}
        for index in topk_indices
    ]

    return {
        "breed": breed,
        "breed_confidence": round(top_probability, 3),
        "species": metadata["species"],
        "size": metadata["size"],
        "size_source": "breed_metadata",
        "color": color,
        "color_confidence": color_confidence,
        "color_source": "dominant_color_heuristic",
        "alternatives": alternatives,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "device": "cpu", "classes": len(classes)}


@app.post("/api/predict")
async def predict_endpoint(file: Optional[UploadFile] = File(default=None)):
    if file is None:
        raise HTTPException(status_code=400, detail="An image file is required")

    try:
        image = Image.open(file.file)
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image")

    return predict(image)


if PWA_DIR.exists():
    app.mount("/assets", StaticFiles(directory=PWA_DIR / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def pwa():
    return FileResponse(PWA_DIR / "index.html")


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest():
    return FileResponse(PWA_DIR / "manifest.webmanifest")
