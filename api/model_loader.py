import os
import joblib

def load_model(model_path="models/HC_RandomForestModel.joblib"):
    full_model_path = os.path.abspath(model_path)

    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model file not found at: {full_model_path}")
    
    try:
        model = joblib.load(full_model_path)
        print(f"✅ Model loaded from {full_model_path}")
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")
