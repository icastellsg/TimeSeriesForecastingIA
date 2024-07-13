import tensorflow as tf
from io import BytesIO
import json
import base64
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
    

def add_day_anual_date(input):     
    day = 60*60*24
    year = 365.2425*day

    seconds = pd.Timestamp.now().timestamp()

    day_sin = np.sin(seconds * (2 * np.pi / day))
    input.append(day_sin)
    day_cos = np.cos(seconds * (2 * np.pi / day))
    input.append(day_cos)
    year_sin = np.sin(seconds * (2 * np.pi / year))
    input.append(year_sin)
    year_cos = np.cos(seconds * (2 * np.pi / year))
    input.append(year_cos)
    return input

def make_predictions_json(input, models, values_to_predict=['temperatura'], scalers=[None]):
    number_labels = len(values_to_predict)
    prediction = []
    
    prediction_1h, _ = get_prediction(models[0], input, "1h", values_to_predict, scalers, plot=False)
    prediction.append(prediction_1h[-1][-number_labels:])
    
    prediction_6h, _ = get_prediction(models[1], input, "6h", values_to_predict, scalers, plot=False)
    prediction.append(prediction_6h[-1][-number_labels:])
    
    prediction_12h, _ = get_prediction(models[2], input, "12h", values_to_predict, scalers, plot=False)
    prediction.append(prediction_12h[-1][-number_labels:])
    
    print(prediction)
    
    for i in range(len(prediction)):
        prediction[i] = np.array(prediction[i], dtype=np.float64).tolist()
    
    return json.dumps(prediction)

def make_predictions(input, models, values_to_predict=['temperatura'], scalers=[None]):
    prediction, graphs = [], []
    number_labels = len(values_to_predict)
    print(input)
    print(number_labels)
    prediction_1h, graphs_1h = get_prediction(models[0], input, "1h", values_to_predict, scalers)
    prediction.append(prediction_1h[-1][-number_labels:])
    for graph in graphs_1h:
        graphs.append(graph)
    
    
    prediction_6h, graphs_6h = get_prediction(models[1], input, "6h", values_to_predict, scalers)
    prediction.append(prediction_6h[-1][-number_labels:])
    for graph in graphs_6h:
        graphs.append(graph)
    
    prediction_12h, graphs_12h = get_prediction(models[2], input, "12h", values_to_predict, scalers)
    prediction.append(prediction_12h[-1][-number_labels:])
    for graph in graphs_12h:
        graphs.append(graph)
    
    return prediction, graphs

def get_prediction(model, value, horizon, values_to_predict=['temperatura'], scalers=[None], plot=True):
    predict = model.predict(np.array([[value]]))
    number_labels = len(values_to_predict)
    graphs = []
    i = 0
    for label in values_to_predict:
        data = predict[:, i::number_labels]
        if scalers[i]:
            data = denormalize_data(data, scalers[i])
            predict[:, i::number_labels] = data
        if plot:
            graphs.append(get_prediction_graph(data, horizon, label))
        i += 1
    return predict, graphs

def get_prediction_graph(prediction, horizon, label):
    fig = Figure()
    ax = fig.subplots()
    ax.set_title("Forecasting " + label + "(horizon: " + str(horizon) + ")")
    ax.set_ylabel("Valor")
    ax.set_xlabel("Minutos")    
    ax.plot(prediction.flatten())
    ax.grid()
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data

def normalize_data(data, scaler):
    aux = scaler.transform([[data]])
    return aux[0][0]
     
def denormalize_data(data, scaler):
    return scaler.inverse_transform(data)