import os

import cv2
import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from polony import predict
from polony.data.utils import highlight_objects

app = FastAPI(title="Polony counter")
predictions_cache = {}
current_file = []


@app.get("/")
def home():
    return "API is working as expected. Now head over to http://localhost:8000/docs."


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    # validate input file
    filename = file.filename
    name, ext = os.path.splitext(filename)
    fileExtension = ext in [".tif", "tif"]
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    predictions = predict(
        path=None,
        file=file,
    )
    global predictions_cache, current_file
    current_file.append(name)
    predictions_cache[name] = predictions[name]
    squares_id_interval = [
        (lambda keys: [min(keys), max(keys)])(predictions[name].keys())
    ]

    return {
        "detail": "Predictions made and stored.",
        "files": current_file,
        "squares_id": squares_id_interval,
    }
