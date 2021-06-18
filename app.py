from flask import Flask, request, render_template
from flask.helpers import get_root_path
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 10*1024*1024

ALLOWED_EXTENSIONS = ['png','jpeg','jpg']

def filetype_check(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def read_input_image(filename):
    img = load_img(filename, color_mode='grayscale', target_size=(28, 28))
    img = img_to_array(img)
    #img = 255-img
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img

@app.route('/', methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['imagefile']
        try:
            if file and filetype_check(file.filename):
                filename = file.filename
                inputfile_path = os.path.join('static/images', filename)
                file.save(inputfile_path)
                img = read_input_image(inputfile_path)
                model = load_model('Apparel_classification_model.h5')
                predicted_class = np.argmax(model.predict(img),axis=-1)
                print(predicted_class)

                
                if predicted_class[0] == 0:
                  Apparel = "T-shirt/top"
                elif predicted_class[0] == 1:
                  Apparel = "Trouser"
                elif predicted_class[0] == 2:
                  Apparel = "Pullover"
                elif predicted_class[0] == 3:
                  Apparel = "Dress"
                elif predicted_class[0] == 4:
                  Apparel = "Coat"
                elif predicted_class[0] == 5:
                  Apparel = "Sandal"
                elif predicted_class[0] == 6:
                  Apparel = "Shirt"
                elif predicted_class[0] == 7:
                  Apparel = "Sneaker"
                elif predicted_class[0] == 8:
                  Apparel = "Bag"
                elif predicted_class[0] == 9:
                  Apparel = "Ankle boot"
                else:
                  Apparel = "Unable to recognise"
                return render_template('predict.html', Apparel = Apparel, input_image = inputfile_path)
        except Exception as e:
            return "Incorrect file format...app supports png, jpg and jpeg file formats only"

    return render_template('predict.html')


if __name__ == "__main__":
    app.run(host='localhost',port=5000,debug=True)


