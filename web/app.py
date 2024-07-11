from flask import Flask, render_template, request
from forms import SuntracerForm
import tensorflow as tf
from io import BytesIO
import json
import base64
from matplotlib.figure import Figure
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'

model1h = load_model('../LSMTTensorflow/bestModel_60.keras')
model6h = load_model('../LSMTTensorflow/bestModel_360.keras')
model12h = load_model('../LSMTTensorflow/bestModel_720.keras')

@app.route('/')
@app.route('/predict/')
@app.route('/knx/')
def index():
    return render_template('predict.html')

@app.route('/suntracer/json', methods=['POST'])
def predict_temperature_suntracer_json():
    temperatura = np.array(request.json['temperatura'])
    prediction = [
        get_prediction(model1h, temperatura)[0][-1],
        get_prediction(model6h, temperatura)[0][-1],
        get_prediction(model12h, temperatura)[0][-1]
    ]
    
    prediction = np.array(prediction, dtype=np.float64).tolist()
    
    return json.dumps(prediction)

@app.route('/suntracer/', methods=['GET','POST'])
def predict_temperature_suntracer():
    form = SuntracerForm()
    if form.validate_on_submit():
        prediction = []
        temperatura = np.array(form.temperatura.data)
        
        prediction.append(get_prediction(model1h, temperatura)[0][-1])
        prediction.append(get_prediction(model6h, temperatura)[0][-1])
        prediction_12h = get_prediction(model12h,temperatura)
        
        prediction.append(prediction_12h[0][-1])
        
        data = get_prediction_graph(prediction_12h)
        
        return render_template('predict_with_flaskForms.html', form=form, prediction=prediction, img_data=data)
    return render_template('predict_with_flaskForms.html', form=form)

def get_prediction(model, value):
    predict = model.predict(np.array([[value]]))
    return predict

def get_prediction_graph(prediction):
    fig = Figure()
    ax = fig.subplots()
    ax.set_title("Forecasting (12h)")
    ax.set_ylabel("Valor")
    ax.set_xlabel("Minutos")
    ax.plot(prediction.flatten())
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data