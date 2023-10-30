import numpy as np

from PIL import Image
from io import BytesIO
# import cv2 as cv

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

# def load_image_from_directory(dir):
#     return cv.imread(dir)