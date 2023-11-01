import os
import uvicorn
import traceback
import numpy as np
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
from utils import load_image_into_numpy_array
import pickle
from sklearn import datasets, svm, metrics
from skimage.feature import local_binary_pattern
from PIL import Image, ImageOps
from fastapi.middleware.cors import CORSMiddleware

# Initialize Model
# If you already put yout model in the same folder as this main.py
# You can load .h5 model or any model below this line

with open('model.pickle', 'rb') as handle:
    model, labels = pickle.load(handle)


# If you use h5 type uncomment line below
# model = tf.keras.models.load_model('./my_model.h5')
# If you use saved model type uncomment line below
# model = tf.saved_model.load("./my_model_folder")


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

def compute_lbp(arr):
    """Find LBP of all pixels.
    Also perform Vectorization/Normalization to get feature vector.
    """
    # LBP function params
    radius = 3
    n_points = 8 * radius
    n_bins = n_points + 2
    lbp = local_binary_pattern(arr, n_points, radius, 'uniform')
    lbp = lbp.ravel()
    feature = np.zeros(n_bins)
    for i in lbp:
        feature[int(i)] += 1
    feature /= np.linalg.norm(feature, ord=1)
    return feature

@app.post("/predict_image")
def predict_image(uploaded_file: UploadFile, response: Response):
    try:
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"

        # image = load_image_into_numpy_array(uploaded_file.file.read())
        img = Image.open(uploaded_file.file)
        if img.mode != 'L':
            img = ImageOps.grayscale(img)
        arr = np.array(img)
        # image = load_image_from_directory('./pura.jpg')
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        # print(image)
        features = compute_lbp(arr)
        # print("Image shape:", image.shape)
        print("Features shape:", features.shape)
        prediction = model.predict(np.array([features]))
        print(prediction)
        print(labels)
        return f"Image is category {labels[prediction[0]]}"
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return f"Internal Server Error: {e}"


port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0', port=port)
