import os
import uvicorn
import traceback
import tensorflow as tf

from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
from utils import load_image_into_numpy_array

# Initialize Model
# If you already put yout model in the same folder as this main.py
# You can load .h5 model or any model below this line

load_options = tf.saved_model.LoadOptions(
    experimental_io_device='/job:localhost')

# If you use h5 type uncomment line below
# model = tf.keras.models.load_model('./my_model.h5')
# If you use saved model type uncomment line below
# model = tf.saved_model.load("./my_model_folder")


app = FastAPI()


@app.get("/")
def index():
    return "Hello world from ML endpoint!"


@app.post("/predict_image")
def predict_image(uploaded_file: UploadFile, response: Response):
    try:
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"

        image = load_image_into_numpy_array(uploaded_file.file.read())
        print("Image shape:", image.shape)

        return "Endpoint not implemented"
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"


port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0', port=port)
