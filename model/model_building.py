"""
Quick Model Training Script
Run this first to generate the model files needed by app.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

print("=" * 60)
print("BREAST CANCER MODEL TRAINING")
print("=" * 60)

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')
    print("✓ Created 'model' directory")

# Load dataset
print("\n1. Loading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target
print(f"   Dataset shape: {df.shape}")

# Select features
selected_features = [
    'mean radius',
    'mean texture', 
    'mean perimeter',
    'mean area',
    'mean compactness'
]

print(f"\n2. Selected features: {selected_features}")

# Prepare data
X = df[selected_features]
y = df['diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n3. Data split:")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")

# Scale features
print("\n4. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\n5. Training Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate
print("\n6. Evaluating model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

# Save model files
print("\n7. Saving model files...")

joblib.dump(model, 'model/breast_cancer_model.pkl')
print("   ✓ Saved: model/breast_cancer_model.pkl")

joblib.dump(scaler, 'model/scaler.pkl')
print("   ✓ Saved: model/scaler.pkl")

joblib.dump(selected_features, 'model/feature_names.pkl')
print("   ✓ Saved: model/feature_names.pkl")

# Test reload
print("\n8. Testing model reload...")
try:
    test_model = joblib.load('model/breast_cancer_model.pkl')
    test_scaler = joblib.load('model/scaler.pkl')
    test_features = joblib.load('model/feature_names.pkl')
    
    # Make a test prediction
    sample = X_test.iloc[0:1]
    sample_scaled = test_scaler.transform(sample)
    prediction = test_model.predict(sample_scaled)[0]
    
    print("   ✓ Model reload successful!")
    print(f"   ✓ Test prediction: {'Benign' if prediction == 1 else 'Malignant'}")
except Exception as e:
    print(f"   ❌ Error reloading model: {e}")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED!")
print("=" * 60)
print("\nYou can now run: python app.py")
print("\nNote: This system is for educational purposes only.")
print("It must not be used as a medical diagnostic tool.")