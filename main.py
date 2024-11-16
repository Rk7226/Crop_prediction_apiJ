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
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and scaler
try:
    model = pickle.load(open('model/random_forest_model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model files: {str(e)}")

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
    return {"status": "healthy"}

@app.post("/predict")
async def predict_crop(data: SoilData):
    """
    Predict the most suitable crop based on soil and climate conditions
    """
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
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)