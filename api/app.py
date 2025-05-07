from flask import Flask, request, jsonify
import numpy as np
from sklearn.metrics import accuracy_score
from api.model_loader import load_model

app = Flask(__name__)
random_forest_model = load_model(model_path="models/HC_RandomForestModel.joblib")
#xg_boost_model_model = load_model(model_path="models/HC_XgBoostModel.joblib")

@app.route('/randomforest/predict', methods=['POST'])
def predict():
    try:

        data = request.json
    
        # Extract input features
        features = np.array(data['input'])

        # Ensure 2D shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Make prediction
        prediction = random_forest_model.predict(features)

        # If expected output is provided, calculate accuracy
        if 'expected_output' in data:
            expected_output = np.array(data['expected_output'])
            accuracy = accuracy_score(expected_output, prediction)
            return jsonify({
                'prediction': prediction.tolist(),
                'expected_output': expected_output.tolist(),
                'accuracy': accuracy
            })

        # If no expected output, return prediction only
        return jsonify({
            'prediction': prediction.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# @app.route('/xgboost/predict', methods=['POST'])
# def predict():
#     try:

#         data = request.json
    
#         # Extract input features
#         features = np.array(data['input'])

#         # Ensure 2D shape
#         if features.ndim == 1:
#             features = features.reshape(1, -1)

#         # Make prediction
#         prediction = random_forest_model.predict(features)

#         # If expected output is provided, calculate accuracy
#         if 'expected_output' in data:
#             expected_output = np.array(data['expected_output'])
#             accuracy = accuracy_score(expected_output, prediction)
#             return jsonify({
#                 'prediction': prediction.tolist(),
#                 'expected_output': expected_output.tolist(),
#                 'accuracy': accuracy
#             })

#         # If no expected output, return prediction only
#         return jsonify({
#             'prediction': prediction.tolist()
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
