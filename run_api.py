# run_api.py
from api.app import app  # Importing the Flask app from API.app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Running the Flask app
