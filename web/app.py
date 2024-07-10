from flask import Flask, render_template, request
from forms import SuntracerForm
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'

model1h = load_model('../LSMTTensorflow/bestModel_60.keras')
model6h = load_model('../LSMTTensorflow/bestModel_360.keras')
model12h = load_model('../LSMTTensorflow/bestModel_720.keras')


@app.route('/')
@app.route('/predict/')
def index():
    return render_template('predict.html')

@app.route('/suntracer/', methods=['GET','POST'])
def predict_temperature_suntracer():
    form = SuntracerForm()
    if form.validate_on_submit():
        prediction = []
        temperatura = np.array(form.temperatura.data)
        
        prediction.append(get_prediction(model1h, temperatura))
        prediction.append(get_prediction(model6h, temperatura))
        prediction.append(get_prediction(model12h, temperatura))
        
        
        return render_template('predict_with_flaskForms.html', form=form, prediction=prediction)
    return render_template('predict_with_flaskForms.html', form=form)

def get_prediction(model, value):
    predict = model.predict(np.array([[value]]))
    return predict[0][-1]