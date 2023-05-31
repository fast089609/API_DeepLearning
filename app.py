from __future__ import division, print_function

# Keras
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input

# Flask 
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import os
import numpy as np
import cv2

import base64
from PIL import Image
from io import BytesIO


width_shape = 224
height_shape = 224

names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# Definimos una instancia de Flask
app = Flask(__name__)

# Path del modelo preentrenado
MODEL_PATH = 'models/model_hojas_VGG16.h5'

# Cargamos el modelo preentrenado
model = load_model(MODEL_PATH)

print('Modelo cargado exitosamente. Verificar http://127.0.0.1:5000/')

# Realizamos la predicción usando la imagen cargada y el modelo
def model_predict(img_path, model):

    img=cv2.resize(cv2.imread(img_path), (width_shape, height_shape), interpolation = cv2.INTER_AREA)
    x=np.asarray(img)
    x=preprocess_input(x)
    x = np.expand_dims(x,axis=0)
    
    preds = model.predict(x)
    return preds


def base64_to_image(base64_data):
    # Dividir la cadena Base64 en nombre y datos
    parts = base64_data.split(',')
    if len(parts) != 2:
        raise ValueError("Formato de cadena Base64 incorrecto. Se esperaba 'nombre,datos'.")

    name = parts[0]
    base64_string = parts[1]

    # Decodificar la cadena Base64 en bytes
    image_bytes = base64.b64decode(base64_string)

    # Crear un objeto BytesIO a partir de los bytes decodificados
    image_buffer = BytesIO(image_bytes)

    # Abrir la imagen desde el buffer
    image = Image.open(image_buffer)

    return name, image


@app.route('/', methods=['GET'])
def index():
    # Página principal
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Obtiene el archivo del re
        # quest
        f = request.files['file']
        image = base64_to_image(f)
        
        # Graba el archivo en ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename('imagen.jpg'))
        f.save(file_path)

        # Predicción
        preds = model_predict(file_path, model)

        print('PREDICCIÓN', names[np.argmax(preds)])
        
        # Enviamos el resultado de la predicción
        result = str(names[np.argmax(preds)])              
        return result
    return None


@app.route('/predict_app', methods=['GET', 'POST'])
def predictapp():
    if request.method == 'POST':

        f = request.form.get('imagen')
        print(f)
        name, image = base64_to_image(f)
        
        # Graba el archivo en ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(name))
        image.save(file_path)

        # Predicción
        preds = model_predict(file_path, model)

        print('PREDICCIÓN', names[np.argmax(preds)])
        
        # Enviamos el resultado de la predicción
        result = str(names[np.argmax(preds)])              
        return result
    return None


if __name__ == '__main__':
    app.run(debug=False, threaded=False)

