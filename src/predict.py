import pickle
import pandas as pd

model = pickle.load(open("models/xgboost_model.pkl","rb"))
scaler = pickle.load(open("models/scaler.pkl","rb"))

def predict(data):

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    return prediction