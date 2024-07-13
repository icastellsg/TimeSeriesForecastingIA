from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, BooleanField, SelectField
from wtforms.validators import DataRequired
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import predictions_helper
from pickle import load

labels = ['Temperatura', 'Brillo', 'Humedad', 'Presión del aire', 'CO2']

model1h_207 = load_model('../LSMTTensorflow/bestModel_sewy_207_60.keras')
model6h_207 = load_model('../LSMTTensorflow/bestModel_sewy_207_360.keras')
model12h_207 = load_model('../LSMTTensorflow/bestModel_sewy_207_720.keras')

scaler_temperatura_207 = load(open('../model_training/scalers/sewy/scaler_temperature_sewy_207.pkl', 'rb'))
scaler_brillo_207 = load(open('../model_training/scalers/sewy/scaler_brightness_sewy_207.pkl', 'rb'))
scaler_humedad_207 = load(open('../model_training/scalers/sewy/scaler_humidity_sewy_207.pkl', 'rb'))
scaler_airpressure_207 = load(open('../model_training/scalers/sewy/scaler_airpressure_sewy_207.pkl', 'rb'))
scaler_co2_207 = load(open('../model_training/scalers/sewy/scaler_co2_sewy_207.pkl', 'rb'))

models_207 = [model1h_207, model6h_207, model12h_207]
scalers_207 = [scaler_temperatura_207, scaler_brillo_207, scaler_humedad_207, scaler_airpressure_207, scaler_co2_207]

model1h_214 = load_model('../LSMTTensorflow/bestModel_sewy_214_60.keras')
model6h_214 = load_model('../LSMTTensorflow/bestModel_sewy_214_360.keras')
model12h_214 = load_model('../LSMTTensorflow/bestModel_sewy_214_720.keras')

scaler_temperatura_214 = load(open('../model_training/scalers/sewy/scaler_temperature_sewy_214.pkl', 'rb'))
scaler_brillo_214 = load(open('../model_training/scalers/sewy/scaler_brightness_sewy_214.pkl', 'rb'))
scaler_humedad_214 = load(open('../model_training/scalers/sewy/scaler_humidity_sewy_214.pkl', 'rb'))
scaler_airpressure_214 = load(open('../model_training/scalers/sewy/scaler_airpressure_sewy_214.pkl', 'rb'))
scaler_co2_214 = load(open('../model_training/scalers/sewy/scaler_co2_sewy_214.pkl', 'rb'))

models_214 = [model1h_214, model6h_214, model12h_214]
scalers_214 = [scaler_temperatura_214, scaler_brillo_214, scaler_humedad_214, scaler_airpressure_214, scaler_co2_214]

model1h_215 = load_model('../LSMTTensorflow/bestModel_sewy_215_60.keras')
model6h_215 = load_model('../LSMTTensorflow/bestModel_sewy_215_360.keras')
model12h_215 = load_model('../LSMTTensorflow/bestModel_sewy_215_720.keras')

scaler_temperatura_215 = load(open('../model_training/scalers/sewy/scaler_temperature_sewy_215.pkl', 'rb'))
scaler_brillo_215 = load(open('../model_training/scalers/sewy/scaler_brightness_sewy_215.pkl', 'rb'))
scaler_humedad_215 = load(open('../model_training/scalers/sewy/scaler_humidity_sewy_215.pkl', 'rb'))
scaler_airpressure_215 = load(open('../model_training/scalers/sewy/scaler_airpressure_sewy_215.pkl', 'rb'))
scaler_co2_215 = load(open('../model_training/scalers/sewy/scaler_co2_sewy_215.pkl', 'rb'))

models_215 = [model1h_215, model6h_215, model12h_215]
scalers_215 = [scaler_temperatura_215, scaler_brillo_215, scaler_humedad_215, scaler_airpressure_215, scaler_co2_215]

class SewyForm(FlaskForm):
    dispositivo = SelectField('Dispositivo', choices=[('207', '207'), ('214', '214'), ('215', '215')], validators=[DataRequired(message='Debe seleccionar un dispositivo')])
    temperatura = FloatField('Temperatura', validators=[DataRequired(message='El valor añadido no es válido')])
    brillo = FloatField('Brillo', validators=[DataRequired(message='El valor añadido no es válido')])
    humedad = FloatField('Humedad', validators=[DataRequired(message='El valor añadido no es válido')])
    presion = FloatField('Presión del aire', validators=[DataRequired(message='El valor añadido no es válido')])
    co2 = FloatField('CO2', validators=[DataRequired(message='El valor añadido no es válido')])
    submit = SubmitField('Predecir')
    

def sewyPredictions():
    form = SewyForm()
    predictions = []
    graphs = []
    if form.validate_on_submit():

        models, scalers = define_model_and_scaler_by_device(form.dispositivo.data)
        
        temperatura = np.array(predictions_helper.normalize_data(form.temperatura.data, scaler=scalers[0]))
        brillo = np.array(predictions_helper.normalize_data(form.brillo.data, scaler=scalers[1]))
        humedad = np.array(predictions_helper.normalize_data(form.humedad.data, scaler=scalers[2]))
        presion = np.array(predictions_helper.normalize_data(form.presion.data, scaler=scalers[3]))
        co2 = np.array(predictions_helper.normalize_data(form.co2.data, scaler=scalers[4]))
        
        input = [temperatura, brillo, humedad, presion, co2]
        input = predictions_helper.add_day_anual_date(input)
        
        predictions, graphs = predictions_helper.make_predictions(input,models, values_to_predict=labels, scalers=scalers)
    return predictions, graphs, labels

def sewyPredictionsJSON(request_as_json):
    
    models, scalers = define_model_and_scaler_by_device(request_as_json['dispositivo'])
    
    temperatura = np.array(predictions_helper.normalize_data(request_as_json['temperatura'], scaler=scalers[0]))
    brillo = np.array(predictions_helper.normalize_data(request_as_json['brillo'], scaler=scalers[1]))
    humedad = np.array(predictions_helper.normalize_data(request_as_json['humedad'], scaler=scalers[2]))
    presion = np.array(predictions_helper.normalize_data(request_as_json['presion'], scaler=scalers[3]))
    co2 = np.array(predictions_helper.normalize_data(request_as_json['co2'], scaler=scalers[4]))
     
    input = [temperatura, brillo, humedad, presion, co2]
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