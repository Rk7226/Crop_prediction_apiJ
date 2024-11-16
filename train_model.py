# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

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
    
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/Crop_recommendation.csv', index=False)
    return df

def train_and_save_model():
    os.makedirs('model', exist_ok=True)
    
    if not os.path.exists('data/Crop_recommendation.csv'):
        df = create_sample_data()
    else:
        df = pd.read_csv('data/Crop_recommendation.csv')
    
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label'].astype('category').cat.codes
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    print("Saving model and scaler...")
    pickle.dump(clf, open('model/random_forest_model.pkl', 'wb'))
    pickle.dump(scaler, open('model/scaler.pkl', 'wb'))
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    train_and_save_model()

# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pickle
from typing import Optional

app = FastAPI(
    title="Crop Prediction API",
    description="API for predicting suitable crops based on soil and climate conditions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler at startup
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        model = pickle.load(open('model/random_forest_model.pkl', 'rb'))
        scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    except Exception as e:
        print(f"Error loading model files: {str(e)}")
        raise RuntimeError("Failed to load model files")

class SoilData(BaseModel):
    N: float = Field(..., description="Nitrogen content in soil", ge=0, le=140)
    P: float = Field(..., description="Phosphorus content in soil", ge=5, le=145)
    K: float = Field(..., description="Potassium content in soil", ge=5, le=205)
    temperature: float = Field(..., description="Temperature in celsius", ge=8.83, le=43.68)
    humidity: float = Field(..., description="Relative humidity in %", ge=14.26, le=99.98)
    ph: float = Field(..., description="pH value of soil", ge=3.5, le=9.94)
    rainfall: float = Field(..., description="Rainfall in mm", ge=20.21, le=298.56)

    class Config:
        schema_extra = {
            "example": {
                "N": 90,
                "P": 42,
                "K": 43,
                "temperature": 20.87,
                "humidity": 82.00,
                "ph": 6.5,
                "rainfall": 202.93
            }
        }

crop_dict = {
    0: "apple", 1: "banana", 2: "blackgram", 3: "chickpea", 4: "coconut",
    5: "coffee", 6: "cotton", 7: "grapes", 8: "jute", 9: "kidneybeans",
    10: "lentil", 11: "maize", 12: "mango", 13: "mothbeans", 14: "mungbean",
    15: "muskmelon", 16: "orange", 17: "papaya", 18: "pigeonpeas", 
    19: "pomegranate", 20: "rice", 21: "watermelon"
}

@app.get("/")
async def root():
    return {
        "message": "Welcome to Crop Prediction API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict")
async def predict_crop(data: SoilData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        features = np.array([
            data.N, data.P, data.K, 
            data.temperature, data.humidity, 
            data.ph, data.rainfall
        ]).reshape(1, -1)
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[0]
        
        crop_name = crop_dict[prediction[0]]
        confidence = float(max(probabilities) * 100)
        
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        alternatives = [
            {
                "crop": crop_dict[idx],
                "confidence": float(probabilities[idx] * 100)
            }
            for idx in top_3_idx
        ]
        
        return {
            "prediction": crop_name,
            "confidence": confidence,
            "alternatives": alternatives,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)