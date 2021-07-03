"""
Run the prediction on Azure machine learning.
"""
import os
import json
import pickle
import numpy as np
from keras.models import load_model


def init():
    """
    Load the model
    """
    global model
    global scaler
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "keras_lenet.h5")
    model = load_model(model_path)
    scaler_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "scaler.pickle")
    with open(scaler_path, "rb") as f_h:
        scaler = pickle.load(f_h)
    f_h.close()


def run(raw_data):
    """
    Prediction
    """
    data = json.loads(raw_data)["data"]
    data = np.array(data)
    data = data.reshape((1, 240, 1))
    # make prediction
    prediction = model.predict(data)
    ans = scaler.inverse_transform(prediction)
    return float(ans)
