import io
import os
import pickle
import uvicorn
import traceback
import numpy as np
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from predict import predict

with open(f'models/label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return "Hello world from ML endpoint!"


@app.post("/predict_image")
def predict_image(uploaded_file: UploadFile, response: Response):
    try:
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"

        # Read the uploaded image file
        img = Image.open(io.BytesIO(uploaded_file.file.read()))

        # Convert the image to grayscale if needed
        img = img.convert("L")  # Convert to grayscale

        # Resize or preprocess the image if required
        # img = img.resize((new_width, new_height))  # Resize image

        # Convert the image to a NumPy array
        img_array = np.array(img)

        prediction = predict(np.array(img), mode='lbp')
        response = f'Prediction : {le.inverse_transform(prediction)[0]}'
        return response
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return f"Internal Server Error: {e}"


port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0', port=port)
