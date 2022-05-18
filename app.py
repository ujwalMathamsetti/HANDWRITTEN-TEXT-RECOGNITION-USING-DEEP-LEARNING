from flask import * 
from flask_cors import CORS
from flask.app import Flask
from flask.templating import render_template
from werkzeug.utils import secure_filename
import os
import plotly.express as px
import pickle
import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os
import random 
import cv2
import imutils
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dense,MaxPooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
import json
from flask import Flask, redirect, url_for
import time

app = Flask(__name__,template_folder='templates')
UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
model_loaded = load_model('pickle/CNN_model.h5')  # to predict the given image results(from saved model of CNN)
LB = pickle.load(open('pickle/CNN_encoder.pkl', 'rb')) # converts image to array format of values

@app.route('/')  
def upload():  
    return render_template("home.html") 


@app.route('/recognition', methods = ['POST'])  
def recognition():  
    if request.method == 'POST':  
        file1 = request.files['img1']  
        print(file1.filename)
        print(UPLOAD_FOLDER + '/' + file1.filename)
        filename1 = secure_filename(file1.filename)
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'],filename1)) 
        imaged = UPLOAD_FOLDER + '/' + file1.filename
        def sort_contours(cnts, method="left-to-right"):
            reverse = False
            i = 0
            if method == "right-to-left" or method == "bottom-to-top":
                reverse = True
            if method == "top-to-bottom" or method == "bottom-to-top":
                i = 1
            boundingBoxes = [cv2.boundingRect(c) for c in cnts]
            (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
            key=lambda b:b[1][i], reverse=reverse))
            # return the list of sorted contours and bounding boxes
            return (cnts, boundingBoxes)
        
        def get_letters(imaged):
            letters = []
            image = cv2.imread(imaged)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
            dilated = cv2.dilate(thresh1, None, iterations=2)
        
            cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sort_contours(cnts, method="left-to-right")[0]
            # loop over the contours
            for c in cnts:
                if cv2.contourArea(c) > 10:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = gray[y:y + h, x:x + w]
                thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_CUBIC)
                thresh = thresh.astype("float32") / 255.0
                thresh = np.expand_dims(thresh, axis=-1)
                thresh = thresh.reshape(1,32,32,1)
                ypred = model_loaded.predict(thresh)
                ypred = LB.inverse_transform(ypred)
                [x] = ypred
                letters.append(x)
            return letters, image
        
        def get_word(letter):
            word = "".join(letter)
            return word
        letter,image = get_letters(imaged)
        word = get_word(letter)
        print(word)
        #plt.imshow(image)
        new_graph_name = "graph" + str(time.time()) + ".png"
        cv2.imwrite('static/'+ new_graph_name, image)
        
        """ret, jpeg = cv2.imencode('.jpg', image)
        jpm = jpeg
        response = make_response(jpeg.tobytes())
        response.headers['Content-Type'] = 'image/png'"""
        print(word)
        #return '<img src=' + url_for('static','result.jpg') + '>' 
        return render_template("result.html",graph = new_graph_name, result = word) 
 

if __name__ == '__main__':  
    app.run()  




       
        