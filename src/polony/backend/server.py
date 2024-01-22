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
    return "API is working as expected. Now head over to http://localhost:8000/docs"


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


@app.post("/highlight")
def highlight(
    filename: str,
    square_id: int,
    highlight_threshold: float = 0.2,
):
    # Check if session_id is in the storage
    if filename not in predictions_cache:
        raise HTTPException(status_code=404, detail="File name not found.")

    # Retrieve predictions for the session
    predictions = predictions_cache[filename]

    # Check if the square_id is valid
    if square_id not in predictions:
        raise HTTPException(status_code=404, detail="Square ID not found.")

    # Get the specific prediction
    predict_dict = predictions[square_id]
    density = predict_dict["density"]
    square = predict_dict["square"]

    # Highlight the selected square
    highlighted_square = highlight_objects(
        original_image=square,
        density_map=density,
        threshold=highlight_threshold,
    )
    dir_name = "resources/images_uploaded"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    img_path = os.path.join(dir_name, f"{filename}-{square_id}.jpeg")
    cv2.imwrite(img_path, highlighted_square)
    file_image = open(img_path, mode="rb")

    # Create JSON-response with additional information
    n_points = predict_dict["result"]
    square_class = predict_dict["class"]
    # json_response = JSONResponse({"n_points": n_points, "square_class": square_class})

    # Create an image streaming response
    headers = {"n_points": str(n_points), "square_class": str(square_class)}
    image_response = StreamingResponse(
        file_image, media_type="image/jpeg", headers=headers
    )

    # Return a response
    return image_response


if __name__ == "__main__":
    # Allows the server to be run in this interactive environment
    nest_asyncio.apply()

    # Host depends on the setup you selected (docker or virtual env)
    host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"

    # Spin up the server!
    uvicorn.run(app, host=host, port=8000)
