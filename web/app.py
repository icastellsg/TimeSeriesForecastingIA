from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

model1h = load_model('../LSMTTensorflow/bestModel_60.keras')
model6h = load_model('../LSMTTensorflow/bestModel_360.keras')
model12h = load_model('../LSMTTensorflow/bestModel_720.keras')


@app.route('/')
def predict():
    return render_template('predict.html')

@app.route('/hello/')
@app.route('/hello/<name>')
def hello_name(name = None):
    return render_template('hello_there.html', name=name)

@app.route('/predict/', methods=['POST'])
def predict_temperature_suntracer():
    prediction = []
    
    input_data = float(request.form['temperatura'])
    predict = model1h.predict(np.array([[input_data]]))
    prediction.append(predict[0][-1])
    predict = model6h.predict(np.array([[input_data]]))
    prediction.append(predict[0][-1])
    predict = model12h.predict(np.array([[input_data]]))
    prediction.append(predict[0][-1])
        
    return render_template('predict.html', prediction=prediction)