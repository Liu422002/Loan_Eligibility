import logging
import joblib


def save_model(model, file_path):
    try:
        joblib.dump(model, file_path)
        logging.info("Model saved successfully to %s", file_path)
    except Exception as e:
        logging.error("Error saving model to %s: %s", file_path, e)
        raise


def load_model(file_path):
    try:
        model = joblib.load(file_path)
        logging.info("Model loaded successfully from %s", file_path)
        return model
    except Exception as e:
        logging.error("Error loading model from %s: %s", file_path, e)
        raise