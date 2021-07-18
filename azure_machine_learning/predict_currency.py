"""
Run the prediction on Azure machine learning.
"""
import os
from datetime import datetime, timedelta
import pickle
from keras.models import load_model
import investpy


def init():
    """
    Load the model
    """
    global model
    global scaler

    for root, _, files in os.walk("/var/azureml-app/azureml-models/", topdown=False):
        for name in files:
            if name.split(".")[-1] == "h5":
                model_path = os.path.join(root, name)
            elif name.split(".")[-1] == "pickle":
                scaler_path = os.path.join(root, name)
    model = load_model(model_path)
    with open(scaler_path, "rb") as f_h:
        scaler = pickle.load(f_h)
    f_h.close()


def run(raw_data):
    """
    Prediction
    """

    today = datetime.now()
    data = investpy.get_currency_cross_historical_data(
        "USD/TWD",
        from_date=(today - timedelta(weeks=105)).strftime("%d/%m/%Y"),
        to_date=today.strftime("%d/%m/%Y"),
    )
    data.reset_index(inplace=True)
    data = data.tail(240).Close.values.reshape(-1, 1)
    data = scaler.fit_transform(data)
    data = data.reshape((1, 240, 1))
    # make prediction
    ans = model.predict(data)
    ans = scaler.inverse_transform(ans)
    return float(ans[0][0])
