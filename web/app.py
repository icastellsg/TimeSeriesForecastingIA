from flask import Flask, render_template, request
from suntracer import SuntracerForm, suntracerPredictions, suntracerPredictionsJSON, labels as suntracer_labels
from sewy import SewyForm, sewyPredictions, sewyPredictionsJSON, labels as sewy_labels
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
        return render_template('predict_with_flaskForms.html', device="Suntracer", form=form, prediction=prediction, img_data=graphs, labels=labels)
    return render_template('predict_with_flaskForms.html', device="Suntracer", form=form)

@app.route('/sewy/json', methods=['POST'])
def predict_temperature_sewy_json():
    prediction = sewyPredictionsJSON(request.json)
    
    return prediction

@app.route('/sewy/', methods=['GET','POST'])
def predict_temperature_sewy():
    form = SewyForm()
    prediction, graphs, labels = sewyPredictions()
    if prediction:
        return render_template('predict_with_flaskForms.html', device="Sewy", form=form, prediction=prediction, img_data=graphs, labels=labels)
    return render_template('predict_with_flaskForms.html', device="Sewy", form=form)



@app.errorhandler(404)
def not_found(e):
    return render_template('404.html')