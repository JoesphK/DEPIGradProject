from flask import Flask, request, jsonify
import numpy as np
from api.model_loader import load_model

app = Flask(__name__)
model = load_model()  # Load from joblib

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.json
        features = np.array(data['input'])

        # Ensure features is 2D (supports both 1D and 2D input)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        prediction = model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
