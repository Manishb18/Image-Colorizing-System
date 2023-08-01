from flask import Flask, render_template, request, url_for
import os
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
from PIL import Image
from pathlib import Path
import datetime
import cv2

global cwd
cwd = Path(__file__).parent.resolve()

def loss(y_true,y_pred) :
    l = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(y_true - y_pred))))
    return l
model_path = cwd / "predictor.h5"
model = load_model(str(model_path), custom_objects={'loss': loss})

app = Flask(__name__)
@app.route('/')
def mainfun():
    return render_template('index.html', img_paths = {'form_disp':'flex','res_disp':'none','original_img':'', 'result_img':''})

@app.route('/predict', methods = ['GET', 'POST'])
def predictor():
    global cwd
    if request.method == 'POST':
        f = request.files['file']
        now = datetime.datetime.now()
        filename = now.strftime("%Y") + now.strftime("%b") + now.strftime("%d") + now.strftime("%H") + now.strftime("%M") + now.strftime("%S") + now.strftime("%f") + ".png"
        file_path = cwd/'static/uploads/'
        file_path = str(file_path) + '/' + filename
        
        f.save(file_path)
        img = keras.utils.load_img(file_path)
        gray_img = keras.utils.img_to_array(img)
        gray_img = resize(gray_img ,(160,160))
        imsave(file_path, gray_img)
        color_me = rgb2lab(1.0/255*gray_img)[:,:,0]
        color_me = color_me.reshape(color_me.shape+(1,))
        
        color_img = model.predict(np.array([color_me]))
        output = color_img[0]
        output = output*128
    
        # Output colorizations
        for i in range(len(output)):
            result = np.zeros((160, 160, 3))
            result[:,:,0] = color_me[:,:,0]
            result[:,:,1:] = output
            color_image = lab2rgb(result)
        # color_image = Image.fromarray((color_image * 255).astype(np.uint8))
        output_fname = cwd/'static/uploads/'
        output_fname = str(output_fname)+'/color'+filename
        imsave(output_fname, color_image)
        dict1 = {'form_disp':'none','res_disp':'flex','original_img':'./static/uploads/'+filename, 'result_img':'./static/uploads/color'+filename}
        return(render_template('index.html', img_paths = dict1))

if __name__ == "__main__":
    app.run(debug=True)