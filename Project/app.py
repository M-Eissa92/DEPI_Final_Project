import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from datetime import datetime
import traceback
import os
import nest_asyncio
import logging
import traceback

# ----------------------- Cell 1: Model + Config -----------------------
PORT = 5000
DEBUG_MODE = True


# Load the trained model and training columns
model_path = "best_forecasting_model_weekly.pkl"
columns_path = "training_columns_weekly.pkl"
input_scaler_path = "input_scaler_weekly.pkl"  # Scaler for input features
target_scaler_path = "target_scaler_weekly.pkl"  # Scaler for the target variable

model = joblib.load(model_path)
training_columns = joblib.load(columns_path)
input_scaler = joblib.load(input_scaler_path)  # Load the input scaler
target_scaler = joblib.load(target_scaler_path)  # Load the target scaler
print("✅ Model and training columns loaded successfully!")

# Initialize Flask app
app = Flask(__name__)

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html', template_folder='templates')

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the raw form data
        print(f"Raw form data: {request.form}")

        # Get input data from the form
        Quantity_Lag1 = float(request.form['Quantity_Lag1'])
        Quantity_Lag4 = float(request.form['Quantity_Lag4'])
        Quantity_Lag8 = float(request.form['Quantity_Lag8'])
        Quantity_Lag12 = float(request.form['Quantity_Lag12'])
        Quantity_Lag52 = float(request.form['Quantity_Lag52'])

        # Log the parsed inputs
        print(f"Parsed inputs - Quantity_Lag1: {Quantity_Lag1}, Quantity_Lag4: {Quantity_Lag4}, "
              f"Quantity_Lag8: {Quantity_Lag8}, Quantity_Lag12: {Quantity_Lag12}, Quantity_Lag52: {Quantity_Lag52}")

        # Create input data as a DataFrame
        input_data = pd.DataFrame([[Quantity_Lag1, Quantity_Lag4, Quantity_Lag8, Quantity_Lag12, Quantity_Lag52]],
                                  columns=training_columns)

        # Log the input DataFrame
        print(f"Input DataFrame: {input_data}")

        # Predict using the loaded model
        prediction = model.predict(input_data)[0]

        # Convert numpy.float32 to Python float
        prediction = float(prediction)

        # Return prediction as JSON response
        return jsonify({'Predicted_Quantity': round(prediction, 2)})
    except KeyError as ke:
        # Handle missing fields in the form data
        return jsonify({'error': f'Missing field: {str(ke)}'}), 400
    except ValueError as ve:
        # Handle invalid input (e.g., non-numeric values)
        return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
    except Exception as e:
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_debugger=True, use_reloader=False)
# ----------------------- Cell 2: Optional Auto Browser Launch -----------------------
#import webbrowser
#webbrowser.open(f"http://127.0.0.1:{PORT}")
