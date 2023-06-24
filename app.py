from __future__ import division, print_function
from keras.utils import load_img, img_to_array

import os
import numpy as np

# Keras
# from keras.applications.imagenet_utils 
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './scripts/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...') 

print('Model loaded. Check http://127.0.0.1:8080/')

def predict_image(image_path):

    # Load the image
    img = load_img(image_path,  target_size=(28, 28), color_mode='grayscale')
    image = img_to_array(img)
    image = image/255
    

    # Predict the label
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

    print(prediction)
    array = np.array(prediction)
    max_index = np.argmax(array)
    return class_labels[max_index]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static', secure_filename(f.filename))
        f.save(file_path)
        
        result = predict_image(file_path)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True ,port=8080)


