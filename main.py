import os
import uvicorn
import traceback
import numpy as np
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import torch

from model.model import GeneratorUNet
from model.utils import ConfigParser

def get_MRI_GAN(pre_trained=True):
    generator = GeneratorUNet()
    if pre_trained:
        # checkpoint_path = ConfigParser.getInstance().get_mri_gan_weight_path()
        # print(ConfigParser.getInstance())
        checkpoint_path = "./DeepFakeDetectModel.chkpt"
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print(checkpoint.keys())
        generator.load_state_dict(checkpoint['model_state_dict'])

    return generator


load_options = tf.saved_model.LoadOptions(
    experimental_io_device='/job:localhost')

model = tf.saved_model.load("./model", options=load_options)

checkpoint_path = "./DeepFakeDetectModel.chkpt"
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_chkpt = YourModelClass().to(device)  # Replace YourModelClass with your actual model class
# model_chkpt.load_state_dict(torch.load(checkpoint_path, map_location=device))

model_chkpt = get_MRI_GAN()
model_chkpt = model_chkpt.to(device)
model_chkpt.eval()


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

@app.post("/predict_image")
def predict_image(uploaded_file: UploadFile, response: Response):
    try:
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"

        img = uploaded_file.file.read()
        img = tf.io.decode_image(img, channels=3)

        image = tf.image.resize(img, [128, 128])
        image = tf.expand_dims(image, axis=0)

        # input_data = image/255.0
      
        result = model(image)
        tensor = tf.constant(result)
        print(result)

        results_array = tensor.numpy()[0].tolist()
        print(results_array)
        return f"Image is category"
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return f"Internal Server Error: {e}"

@app.post("/predict_image_chkpt")
def predict_image_chkpt(uploaded_file: UploadFile, response: Response):
    try:
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"

        img = uploaded_file.file.read()
        img = tf.io.decode_image(img, channels=3)

        image = tf.image.resize(img, [128, 128])
        image = tf.expand_dims(image, axis=0)

        with torch.no_grad():
            output = model(image)

        # result = process_output(output)

        # input_data = image/255.0
      
        # result = model(image)
        # tensor = tf.constant(result)
        # print(result)
        print(output)

        # results_array = tensor.numpy()[0].tolist()
        # print(results_array)
        return f"Image is category"
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return f"Internal Server Error: {e}"


port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0', port=port)
