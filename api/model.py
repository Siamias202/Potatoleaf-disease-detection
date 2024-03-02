from flask import Flask ,render_template,request
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model









class_name =[
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight"
]

saved_model = load_model("potato-disease-detection-model.h5",compile=False)

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
       
        
        img =Image.open(self.filename).resize((256, 256), Image.LANCZOS)
        img_np = np.asarray(img).astype('uint8')
        img_tf = tf.expand_dims(img_np, 0)
        prediction=saved_model.predict(img_tf)
        predicted_class = class_name[np.argmax(prediction)]
        return predicted_class

# def predict():
#     imagefile = request.files['imagefile']
#     image_path="./images/"+imagefile.filename
#     imagefile.save(image_path)

#     img=Image.open(image_path)
#     target_size=(256,256)
#     img=img.resize(target_size,Image.LANCZOS)
#     img_np = np.asarray(img).astype('uint8')
#     img_tf = tf.expand_dims(img_np, 0)
#     prediction=saved_model.predict(img_tf)
#     predicted_class = class_name[np.argmax(prediction)]
#     return predicted_class

   


