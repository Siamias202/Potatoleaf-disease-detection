from flask import Flask ,render_template,request
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

app=Flask(__name__)



@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')



class_name =[
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight"
]

saved_model = load_model("potato-disease-detection-model.h5",compile=False)

@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path="./images/"+imagefile.filename
    imagefile.save(image_path)

    img=Image.open(image_path)
    target_size=(256,256)
    img=img.resize(target_size,Image.LANCZOS)
    img_np = np.asarray(img).astype('uint8')
    img_tf = tf.expand_dims(img_np, 0)
    prediction=saved_model.predict(img_tf)
    predicted_class = class_name[np.argmax(prediction)]

    return render_template('index.html',prediction=predicted_class)



if __name__=='__main__':
    app.run(port=3000,debug=True)