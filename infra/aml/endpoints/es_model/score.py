import json
import pickle
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    logger.info(f"Model directory: {model_dir}")

    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pkl not found in {model_dir}")

    logger.info(f"Loading model from: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded successfully: {type(model)}")

def run(raw_data):
    data = json.loads(raw_data)
    features = np.array(data["features"])
    if features.ndim == 1:
        features = features.reshape(1, -1)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    return json.dumps({
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    })
