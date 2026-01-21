"""
Breast Cancer Prediction System - Web Application
Part B: Web GUI Application using Flask
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model, scaler, and feature names
# Try multiple possible paths
def load_models():
    possible_paths = [
        ('model/breast_cancer_model.pkl', 'model/scaler.pkl', 'model/feature_names.pkl'),
        ('breast_cancer_model.pkl', 'scaler.pkl', 'feature_names.pkl'),
        ('./model/breast_cancer_model.pkl', './model/scaler.pkl', './model/feature_names.pkl'),
    ]
    
    for model_path, scaler_path, features_path in possible_paths:
        try:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                feature_names = joblib.load(features_path)
                print(f"✓ Model loaded successfully from: {model_path}")
                return model, scaler, feature_names
        except Exception as e:
            continue
    
    print("❌ Error: Could not find model files!")
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    if os.path.exists('model'):
        print("Files in model directory:", os.listdir('model'))
    return None, None, None

model, scaler, feature_names = load_models()

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please check server configuration.'
            }), 500
        
        # Get input data from form
        features_data = []
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None or value == '':
                return jsonify({
                    'error': f'Missing value for {feature}'
                }), 400
            features_data.append(float(value))
        
        # Convert to numpy array and reshape
        input_array = np.array(features_data).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Prepare result
        result = {
            'prediction': 'Benign' if prediction == 1 else 'Malignant',
            'prediction_class': int(prediction),
            'confidence': {
                'malignant': f"{probability[0]*100:.2f}%",
                'benign': f"{probability[1]*100:.2f}%"
            },
            'features': dict(zip(feature_names, features_data))
        }
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({
            'error': f'Invalid input: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features': feature_names
    })

if __name__ == '__main__':
    # For development
    app.run(debug=True, host='0.0.0.0', port=5000)