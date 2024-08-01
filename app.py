from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Carga tu modelo
MODEL_PATH = 'https://drive.google.com/file/d/1GcnmKQXQL8kpiEzhUsy_MjjwQxEvzTt4/view?usp=drive_link'
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', f.filename)
        f.save(file_path)
        prediction = model_predict(file_path, model)
        result = 'Dog' if prediction[0] > 0.5 else 'Cat'
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)