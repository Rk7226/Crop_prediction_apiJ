import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Sample data for demonstration (you should replace this with your actual data)
def create_sample_data():
    # Creating sample data
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
        'label': np.random.randint(0, 22, n_samples)  # 22 different crops
    }
    
    df = pd.DataFrame(data)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/Crop_recommendation.csv', index=False)
    return df

def train_and_save_model():
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Check if data exists, if not create sample data
    if not os.path.exists('data/Crop_recommendation.csv'):
        df = create_sample_data()
    else:
        df = pd.read_csv('data/Crop_recommendation.csv')
    
    # Prepare features and target
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label'].astype('category').cat.codes
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    print("Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    print("Saving model and scaler...")
    pickle.dump(clf, open('model/random_forest_model.pkl', 'wb'))
    pickle.dump(scaler, open('model/scaler.pkl', 'wb'))
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    train_and_save_model()