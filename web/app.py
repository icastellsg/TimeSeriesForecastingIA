from flask import Flask, render_template, request
from suntracer import SuntracerForm, suntracerPredictions, suntracerPredictionsJSON, labels as suntracer_labels
import json
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'

@app.route('/')
@app.route('/predict/')
@app.route('/knx/')
def index():
    return render_template('predict.html')

@app.route('/suntracer/json', methods=['POST'])
def predict_temperature_suntracer_json():
    prediction = suntracerPredictionsJSON(request.json)
    
    return prediction

@app.route('/suntracer/', methods=['GET','POST'])
def predict_temperature_suntracer():
    form = SuntracerForm()
    prediction, graphs, labels = suntracerPredictions()
    if prediction:
        return render_template('predict_with_flaskForms.html', form=form, prediction=prediction, img_data=graphs, labels=labels)
    return render_template('predict_with_flaskForms.html', form=form)

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html')