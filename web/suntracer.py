from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, BooleanField
from wtforms.validators import DataRequired
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import predictions_helper
from pickle import load

model1h = load_model('../LSMTTensorflow/bestModel_MultiVariable_60.keras')
model6h = load_model('../LSMTTensorflow/bestModel_MultiVariable_360.keras')
model12h = load_model('../LSMTTensorflow/bestModel_MultiVariable_720.keras')

scaler_temperatura = load(open('../model_training/scalers/suntracer/scaler_temperature_suntracer.pkl', 'rb'))
scaler_brillo = load(open('../model_training/scalers/suntracer/scaler_brightness_suntracer.pkl', 'rb'))
scaler_viento = load(open('../model_training/scalers/suntracer/scaler_wind_suntracer.pkl', 'rb'))

models = [model1h, model6h, model12h]
labels = ['Temperatura', 'Brillo', 'Viento', 'Lluvia']
scalers = [scaler_temperatura, scaler_brillo, scaler_viento, None]

class SuntracerForm(FlaskForm):
    temperatura = FloatField('Temperatura', validators=[DataRequired(message='El valor añadido no es válido')])
    brillo = FloatField('Brillo', validators=[DataRequired(message='El valor añadido no es válido')])
    viento = FloatField('Viento', validators=[DataRequired(message='El valor añadido no es válido')])
    lluvia = BooleanField('Lluvia')
    submit = SubmitField('Predecir')

def suntracerPredictions():
    form = SuntracerForm()
    predictions = []
    graphs = []
    if form.validate_on_submit():
        
        temperatura = np.array(predictions_helper.normalize_data(form.temperatura.data, scaler=scaler_temperatura))
        brillo = np.array(predictions_helper.normalize_data(form.brillo.data, scaler=scaler_brillo))
        viento = np.array(predictions_helper.normalize_data(form.viento.data, scaler=scaler_viento))
        lluvia = np.array(form.lluvia.data)
        
        input = [temperatura, brillo, viento, lluvia]
        input = predictions_helper.add_day_anual_date(input)
        
        predictions, graphs = predictions_helper.make_predictions(input,models, values_to_predict=labels, scalers=scalers)
    return predictions, graphs, labels, form

def suntracerPredictionsJSON(request_as_json):
    temperatura = np.array(predictions_helper.normalize_data(request_as_json['temperatura'], scaler=scaler_temperatura))
    brillo = np.array(predictions_helper.normalize_data(request_as_json['brillo'], scaler=scaler_brillo))
    viento = np.array(predictions_helper.normalize_data(request_as_json['viento'], scaler=scaler_viento))
    lluvia = np.array(request_as_json['lluvia'])
    
    input = [temperatura, brillo, viento, lluvia]
    input = predictions_helper.add_day_anual_date(input)
    
    return predictions_helper.make_predictions_json(input,models, values_to_predict=labels, scalers=scalers)