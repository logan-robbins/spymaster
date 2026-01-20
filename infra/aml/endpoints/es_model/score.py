import json
import pickle
import os
import numpy as np
import logging
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    logger.info(f"Model directory: {model_dir}")

    # List all contents for debugging
    all_files = []
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            full_path = os.path.join(root, f)
            all_files.append(full_path)
            logger.info(f"Found file: {full_path}")

    # Try to find model_out file (the pickle file saved by training)
    model_path = None
    for f in all_files:
        if f.endswith("model_out") or "model" in os.path.basename(f).lower():
            model_path = f
            logger.info(f"Selected model file: {model_path}")
            break

    # If still not found, try the first file
    if model_path is None and all_files:
        model_path = all_files[0]
        logger.info(f"Fallback to first file: {model_path}")

    if model_path is None:
        raise FileNotFoundError(f"Could not find model file in {model_dir}. Contents: {all_files}")

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
