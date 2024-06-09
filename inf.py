import keras
import sys, os
import numpy as np
from PIL import Image
from keras.models import load_model

imsize = (64, 64)

testpic = "./image/img_snoopy.png"
keras_param = "cnn.h5"

def load_image(path):
    img = Image.open(path)
    img = img.convert("RGB")  # RGBに変換
    img = img.resize(imsize)
    img = np.asarray(img)
    img = img / 255.0
    return img 

model = load_model(keras_param)
img = load_image(testpic)
prd = model.predict(np.array([img]))
print(prd)

prelabel = np.argmax(prd, axis=1)
if prelabel == 0:
    print(">>> 犬")
elif prelabel == 1:
    print(">>> 猫")