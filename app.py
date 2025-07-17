import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import os
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load trained xgboost model 
model = joblib.load("xgboost_model.pkl")
gender_enc = joblib.load('gender_encoder.pkl')
married_enc = joblib.load('ever_married_encoder.pkl')
work_enc = joblib.load('work_type_encoder.pkl')
res_enc = joblib.load('Residence_type_encoder.pkl')
smoke_enc = joblib.load('smoking_status_encoder.pkl')
scaler = joblib.load('scaler.pkl')

@app.route("/", methods=["GET"])
def run():
    return "Flask API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symptoms = data.get('symptoms', None)
        input = pd.DataFrame([symptoms])

        input['hypertension'] = input['hypertension'].astype(int)
        input['heart_disease'] = input['heart_disease'].astype(int)

        # Encode using the encoder used in the training set
        input['gender'] = gender_enc.transform(input['gender'])
        input['ever_married'] = married_enc.transform(input['ever_married'])
        input['work_type'] = work_enc.transform(input['work_type'])
        input['Residence_type'] = res_enc.transform(input['Residence_type'])
        input['smoking_status'] = smoke_enc.transform(input['smoking_status'])

        input[['age', 'avg_glucose_level', 'bmi']] = scaler.transform(
        input[['age', 'avg_glucose_level', 'bmi']]
        )

        # Make prediction
        prediction = model.predict(input)[0]
        proba = model.predict_proba(input)[0][1]

        # Interpret result
        if prediction == 1:
            result = "Based on your responses, there are signs that may indicate a stroke risk. Please consult a healthcare professional."
        else:
            result = "Your responses do not currently suggest a high stroke risk. Stay healthy and check in regularly!"

        return jsonify({
            "prediction": result,
            "probability": round(float(proba), 4)
        })
    except Exception as e:
        return jsonify({'error': "Error during prediction"}), 400

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))