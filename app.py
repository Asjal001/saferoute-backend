from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Allow React to communicate with this server

# 1. LOAD YOUR MODEL
print("Loading SafeRoute Brain...")
model = load_model('SafeRoute_Model.h5')
preprocessor = joblib.load('preprocessor.pkl')
print("System Ready!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. RECEIVE DATA FROM REACT
        data = request.json
        print("Received Input:", data)
        
        # 3. PREPARE DATA FRAME
        # We must match the EXACT columns used during training
        input_df = pd.DataFrame([{
            'Hour': int(data['hour']),
            'DayOfWeek': int(data['day']),
            'Latitude': float(data['lat']),
            'Longitude': float(data['lon']),
            'Avg_Speed(km/h)': float(data['speed']),
            'Road_ID': data['road'],
            'Weather': data['weather']
        }])

        # 4. PREPROCESS (Scale numbers, encode text)
        processed_input = preprocessor.transform(input_df)
        
        # Neural Network expects a specific array shape
        if hasattr(processed_input, "toarray"):
            processed_input = processed_input.toarray()

        # 5. PREDICT
        # The model returns a list: [vehicle_prediction, accident_prediction]
        predictions = model.predict(processed_input)
        
        vehicle_count = int(predictions[0][0][0])
        accident_prob = float(predictions[1][0][0])

        # 6. GENERATE INTELLIGENT LABELS
        
        # Traffic Density Logic
        density_label = "Low"
        if vehicle_count > 200: density_label = "Moderate"
        if vehicle_count > 300: density_label = "High"
        if vehicle_count > 400: density_label = "Severe Congestion"

        # Accident Risk Logic
        risk_label = "Safe"
        if accident_prob > 0.3: risk_label = "Caution"
        if accident_prob > 0.5: risk_label = "Danger"
        if accident_prob > 0.8: risk_label = "High Risk Area"

        # 7. SEND RESPONSE
        response = {
            'vehicle_count': vehicle_count,
            'traffic_density': density_label,
            'accident_likelihood': round(accident_prob * 100, 1), # Percentage
            'risk_label': risk_label
        }
        return jsonify(response)

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)