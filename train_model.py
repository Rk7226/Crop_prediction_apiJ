import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def create_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'N': np.random.uniform(0, 140, n_samples),
        'P': np.random.uniform(5, 145, n_samples),
        'K': np.random.uniform(5, 205, n_samples),
        'temperature': np.random.uniform(8.83, 43.68, n_samples),
        'humidity': np.random.uniform(14.26, 99.98, n_samples),
        'ph': np.random.uniform(3.5, 9.94, n_samples),
        'rainfall': np.random.uniform(20.21, 298.56, n_samples),
        'label': np.random.randint(0, 22, n_samples)
    }
    
    return pd.DataFrame(data)

def train_and_save_model():
    print("Starting model training...")
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Create sample data
    df = create_sample_data()
    
    # Prepare features and target
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model
    print("Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Save the model and scaler using joblib
    print("Saving model and scaler...")
    model_path = os.path.join('model', 'random_forest_model.pkl')
    scaler_path = os.path.join('model', 'scaler.pkl')
    
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    train_and_save_model()