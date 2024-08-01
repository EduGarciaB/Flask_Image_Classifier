from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
#from werkzeug.utils import secure_filename
import tensorflow as tf


app = Flask(__name__)

# Ruta al modelo
MODEL_PATH = '/Users/luiseduardogarciablanco/Desktop/bootcamp/flask_image_classifier/best_model.h5'
model = load_model(MODEL_PATH)

def model_predict(file_path, model):
    # Carga y preprocesa la imagen
    img = load_img(file_path, target_size=(100, 100))  # Cambiar a (100, 100)
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Añadir dimensión batch
    prediction = model.predict(img_array)
    return prediction

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(upload_path)

        prediction = model_predict(upload_path, model)
        # Asumiendo que prediction es un array con una sola predicción
        result = 'Dog' if prediction[0][0] > 0.5 else 'Cat'

        # Generar HTML para mostrar la imagen y el resultado
        img_tag = f'<img src="/uploads/{f.filename}" style="max-width: 300px;"><br>'
        result_html = f'<p>Prediction: {result}</p><br>'
        retry_button = '<button onclick="window.location.reload();">Prueba de Nuevo</button>'
        return img_tag + result_html + retry_button

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)

if __name__ == '__main__':
    app.run(debug=True)