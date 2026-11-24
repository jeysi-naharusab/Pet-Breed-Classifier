# Pet Breed Classifier PWA

This repository serves a FastAPI prediction API and installable PWA alongside the existing EfficientNet-B0 breed model.

## Run locally

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Upload a photo and select **Detect Breed**.

## Deploy to Render

1. Push the repository to GitHub. The ONNX model and class file must be committed:

```powershell
git add .python-version render.yaml requirements.txt api.py .gitignore README.md
git add checkpoints/classes.json checkpoints/pet-breed-classifier.onnx checkpoints/pet-breed-classifier.onnx.data
git commit -m "Configure Render deployment"
git push origin main
```

2. In Render, choose **New > Blueprint** and select this repository. Accept the service named `pet-breed-classifier`, or use **New > Web Service** and enter:

   - Runtime: **Python**
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/api/health`

3. Deploy. The PWA and API will be available at the Render service URL, for example `https://pet-breed-classifier.onrender.com`.

The free plan may spin down after inactivity and the first request may take longer while the service starts. Render requires the model files to be inside the Git repository (not only in a local ignored directory).

## API

`GET /api/health` reports API status, runtime device, and class count.

`POST /api/predict` accepts a multipart image field named `file` and returns:

```json
{
  "breed": "Golden_Retriever",
  "breed_confidence": 0.92,
  "species": "Dog",
  "size": "Large",
  "size_source": "breed_metadata",
  "color": "Golden",
  "color_confidence": 0.61,
  "color_source": "dominant_color_heuristic",
  "alternatives": []
}
```

Species and size come from `breed_metadata.json`. Size is breed-derived and is not an individual-animal measurement. Color is a coarse dominant-color estimate and should be labeled as an estimate in the UI.

## Export ONNX

```powershell
python src\export.py
```

This exports `checkpoints/best.pt` to `checkpoints/pet-breed-classifier.onnx` and preserves the class order from `checkpoints/classes.json`.
