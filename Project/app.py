from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import xgboost
# Initialize Flask app
app = Flask(__name__)

# Load the trained model and training columns
model_path = "best_forecasting_model.pkl"
columns_path = "training_columns.pkl"

model = joblib.load(model_path)
training_columns = joblib.load(columns_path)

print("✅ Model and training columns loaded successfully!")

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        Quantity_Lag1 = float(request.form['Quantity_Lag1'])
        Quantity_Lag7 = float(request.form['Quantity_Lag7'])

        # Create input data as a DataFrame
        input_data = pd.DataFrame([[Quantity_Lag1, Quantity_Lag7]], columns=training_columns)

        # Predict using the loaded model
        prediction = model.predict(input_data)[0]

        # Convert numpy.float32 to Python float
        prediction = float(prediction)

        # Return prediction as JSON response
        return jsonify({'Predicted_Sales': round(prediction, 2)})
    except Exception as e:
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    print("🚀 Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)