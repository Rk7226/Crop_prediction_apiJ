import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import uvicorn
from typing import Optional

# Create FastAPI app
app = FastAPI(
    title="Crop Prediction API",
    description="API for predicting suitable crops based on soil and climate conditions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
model = None
scaler = None

# Input validation model
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

# Crop dictionary
crop_dict = {
    0: "apple", 1: "banana", 2: "blackgram", 3: "chickpea", 4: "coconut",
    5: "coffee", 6: "cotton", 7: "grapes", 8: "jute", 9: "kidneybeans",
    10: "lentil", 11: "maize", 12: "mango", 13: "mothbeans", 14: "mungbean",
    15: "muskmelon", 16: "orange", 17: "papaya", 18: "pigeonpeas", 
    19: "pomegranate", 20: "rice", 21: "watermelon"
}

@app.on_event("startup")
async def load_model():
    """Load the model and scaler on startup"""
    global model, scaler
    try:
        print("Loading model files...")
        model_path = os.path.join(os.path.dirname(__file__), "model", "random_forest_model.pkl")
        scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")
        
        print(f"Model path: {model_path}")
        print(f"Scaler path: {scaler_path}")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("Model files not found, training new model...")
            from train_model import train_and_save_model
            train_and_save_model()
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading model files: {str(e)}")

@app.get("/")
async def root():
    """Welcome endpoint with API information"""
    return {
        "message": "Welcome to Crop Prediction API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Model or scaler not loaded. Please ensure model files exist in the correct location."
        )
    return {"status": "healthy"}

@app.post("/predict")
async def predict_crop(data: SoilData):
    """Predict the most suitable crop based on soil and climate conditions"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Model or scaler not loaded. Please check health endpoint for status."
        )
        
    try:
        # Convert input data to array
        features = np.array([
            data.N, data.P, data.K, 
            data.temperature, data.humidity, 
            data.ph, data.rainfall
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        # Get crop name
        crop_name = crop_dict[prediction[0]]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities) * 100)
        
        # Get top 3 predictions
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
        raise HTTPException(
            status_code=500, 
            detail=f"Error making prediction: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)