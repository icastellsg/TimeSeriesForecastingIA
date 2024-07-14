from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, SelectField
from wtforms.validators import DataRequired
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import predictions_helper
from pickle import load

labels = ['Temperatura', 'Humedad', 'CO2']

model1h_207 = load_model('../LSMTTensorflow/bestModel_touch_207_60.keras')
model6h_207 = load_model('../LSMTTensorflow/bestModel_touch_207_360.keras')
model12h_207 = load_model('../LSMTTensorflow/bestModel_touch_207_720.keras')

scaler_temperatura_207 = load(open('../model_training/scalers/touch/scaler_temperature_touch_207.pkl', 'rb'))
scaler_humedad_207 = load(open('../model_training/scalers/touch/scaler_humidity_touch_207.pkl', 'rb'))
scaler_co2_207 = load(open('../model_training/scalers/touch/scaler_co2_touch_207.pkl', 'rb'))

models_207 = [model1h_207, model6h_207, model12h_207]
scalers_207 = [scaler_temperatura_207, scaler_humedad_207, scaler_co2_207]

model1h_214 = load_model('../LSMTTensorflow/bestModel_touch_214_60.keras')
model6h_214 = load_model('../LSMTTensorflow/bestModel_touch_214_360.keras')
model12h_214 = load_model('../LSMTTensorflow/bestModel_touch_214_720.keras')

scaler_temperatura_214 = load(open('../model_training/scalers/touch/scaler_temperature_touch_214.pkl', 'rb'))
scaler_humedad_214 = load(open('../model_training/scalers/touch/scaler_humidity_touch_214.pkl', 'rb'))
scaler_co2_214 = load(open('../model_training/scalers/touch/scaler_co2_touch_214.pkl', 'rb'))

models_214 = [model1h_214, model6h_214, model12h_214]
scalers_214 = [scaler_temperatura_214, scaler_humedad_214, scaler_co2_214]

model1h_215 = load_model('../LSMTTensorflow/bestModel_touch_215_60.keras')
model6h_215 = load_model('../LSMTTensorflow/bestModel_touch_215_360.keras')
model12h_215 = load_model('../LSMTTensorflow/bestModel_touch_215_720.keras')

scaler_temperatura_215 = load(open('../model_training/scalers/touch/scaler_temperature_touch_215.pkl', 'rb'))
scaler_humedad_215 = load(open('../model_training/scalers/touch/scaler_humidity_touch_215.pkl', 'rb'))
scaler_co2_215 = load(open('../model_training/scalers/touch/scaler_co2_touch_215.pkl', 'rb'))

models_215 = [model1h_215, model6h_215, model12h_215]
scalers_215 = [scaler_temperatura_215, scaler_humedad_215, scaler_co2_215]

class TouchForm(FlaskForm):
    dispositivo = SelectField('Dispositivo', choices=[('207', '207'), ('214', '214'), ('215', '215')], validators=[DataRequired(message='Debe seleccionar un dispositivo')])
    temperatura = FloatField('Temperatura', validators=[DataRequired(message='El valor añadido no es válido')])
    humedad = FloatField('Humedad', validators=[DataRequired(message='El valor añadido no es válido')])
    co2 = FloatField('CO2', validators=[DataRequired(message='El valor añadido no es válido')])
    submit = SubmitField('Predecir')
    

def touchPredictions():
    form = TouchForm()
    predictions = []
    graphs = []
    if form.validate_on_submit():

        models, scalers = define_model_and_scaler_by_device(form.dispositivo.data)
        
        temperatura = np.array(predictions_helper.normalize_data(form.temperatura.data, scaler=scalers[0]))
        humedad = np.array(predictions_helper.normalize_data(form.humedad.data, scaler=scalers[1]))
        co2 = np.array(predictions_helper.normalize_data(form.co2.data, scaler=scalers[2]))
        
        input = [temperatura, humedad, co2]
        input = predictions_helper.add_day_anual_date(input)
        
        predictions, graphs = predictions_helper.make_predictions(input,models, values_to_predict=labels, scalers=scalers)
    return predictions, graphs, labels, form

def touchPredictionsJSON(request_as_json):
    
    models, scalers = define_model_and_scaler_by_device(request_as_json['dispositivo'])
    
    temperatura = np.array(predictions_helper.normalize_data(request_as_json['temperatura'], scaler=scalers[0]))
    humedad = np.array(predictions_helper.normalize_data(request_as_json['humedad'], scaler=scalers[1]))
    co2 = np.array(predictions_helper.normalize_data(request_as_json['co2'], scaler=scalers[2]))
     
    input = [temperatura, humedad, co2]
    input = predictions_helper.add_day_anual_date(input)
    
    return predictions_helper.make_predictions_json(input,models, values_to_predict=labels, scalers=scalers)

def define_model_and_scaler_by_device(device):
    if device == '207':
        return models_207, scalers_207
    elif device == '214':
        return models_214, scalers_214
    elif device == '215':
        return models_215, scalers_215
    else:
        return None, None