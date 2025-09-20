from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import json
import os
from typing import Literal, Dict, Any

# --- 1. Initialize the FastAPI App ---
app = FastAPI(
    title="Asteroid Risk Prediction API",
    description="An API that uses a RandomForest model trained on your real JSON data to predict NEO risk.",
    version="1.0.0"
)

# --- 2. Load the Trained Model and Labels ---
try:
    model = joblib.load('asteroid_risk_model.joblib')
    label_map = joblib.load('risk_level_labels.joblib')
    
    # Try to load training metadata if available
    try:
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        TRAINING_FEATURES = metadata.get('features_used', [
            'distance_au', 'velocity_kms', 'diameter_km', 'v_infinity_kms', 'is_pha',
            'class_AMO', 'class_APO', 'class_ATE', 'class_IEO'
        ])
        print(f"Model loaded successfully. Trained on {metadata.get('training_records', 'unknown')} records.")
    except FileNotFoundError:
        # Fallback to default features
        TRAINING_FEATURES = [
            'distance_au', 'velocity_kms', 'diameter_km', 'v_infinity_kms', 'is_pha',
            'class_AMO', 'class_APO', 'class_ATE', 'class_IEO'
        ]
        print("Model loaded successfully (without metadata).")
        
except FileNotFoundError as e:
    print(f"ERROR: Model files not found - {e}")
    print("Please run `python train_model.py` first with your `real_asteroid_data.json` file.")
    model = None
    label_map = None
    TRAINING_FEATURES = []


# --- 3. Define the Input Data Model using Pydantic ---
class AsteroidFeatures(BaseModel):
    distance_au: float = Field(
        ..., 
        example=0.018, 
        description="Closest approach distance in Astronomical Units.",
        gt=0
    )
    velocity_kms: float = Field(
        ..., 
        example=22.3, 
        description="Relative velocity in km/s.",
        gt=0
    )
    diameter_km: float = Field(
        ..., 
        example=0.45, 
        description="Estimated diameter in km.",
        gt=0
    )
    v_infinity_kms: float = Field(
        ..., 
        example=18.1, 
        description="Velocity at infinity in km/s.",
        gt=0
    )
    is_pha: bool = Field(
        ..., 
        example=True, 
        description="Is the object a Potentially Hazardous Asteroid?"
    )
    orbit_class: Literal['ATE', 'APO', 'AMO', 'IEO'] = Field(
        ..., 
        example='APO', 
        description="The asteroid's orbit class."
    )

    class Config:
        schema_extra = {
            "example": {
                "distance_au": 0.018,
                "velocity_kms": 22.3,
                "diameter_km": 0.45,
                "v_infinity_kms": 18.1,
                "is_pha": True,
                "orbit_class": "APO"
            }
        }


class PredictionResponse(BaseModel):
    input_features: Dict[str, Any]
    predicted_risk_level: str
    predicted_risk_score: float
    confidence: float
    prediction_probabilities: Dict[str, float]
    model_info: Dict[str, Any]


# --- Helper function for risk score calculation ---
def calculate_risk_score(probabilities: np.ndarray) -> float:
    """Calculate a weighted risk score from probabilities."""
    weights = np.array([0, 0.33, 0.66, 1.0])  # Low, Medium, High, Critical
    padded_probs = np.zeros(4)
    padded_probs[:len(probabilities)] = probabilities
    return float(np.dot(padded_probs, weights))


def get_confidence_score(probabilities: np.ndarray) -> float:
    """Calculate confidence as the maximum probability."""
    return float(np.max(probabilities))


# --- 4. Create the Prediction Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
def predict_risk(asteroid: AsteroidFeatures):
    """
    Predict the risk level of an asteroid based on its orbital and physical characteristics.
    
    The model was trained on real asteroid data in JSON format and uses a RandomForest classifier
    to predict risk levels: Low, Medium, High, or Critical.
    """
    if not model or not label_map:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server startup logs and ensure the model files exist."
        )

    try:
        # Convert input to DataFrame
        input_df_raw = pd.DataFrame([asteroid.dict()])
        
        # Apply one-hot encoding for orbit_class (same as training)
        input_df_processed = pd.get_dummies(input_df_raw, columns=['orbit_class'], prefix='class')
        
        # Ensure all training features are present
        input_df_final = input_df_processed.reindex(columns=TRAINING_FEATURES, fill_value=0)
        
        # Make predictions
        prediction_encoded = model.predict(input_df_final)[0]
        probabilities = model.predict_proba(input_df_final)[0]
        predicted_level = label_map[prediction_encoded]
        risk_score = calculate_risk_score(probabilities)
        confidence = get_confidence_score(probabilities)
        
        # Format probability results
        prob_dict = {
            label_map.get(i, f"Unknown_{i}"): round(float(p), 4) 
            for i, p in enumerate(probabilities)
        }
        
        return PredictionResponse(
            input_features=asteroid.dict(),
            predicted_risk_level=predicted_level,
            predicted_risk_score=round(risk_score, 4),
            confidence=round(confidence, 4),
            prediction_probabilities=prob_dict,
            model_info={
                "model_type": "RandomForestClassifier",
                "features_used": len(TRAINING_FEATURES),
                "available_classes": list(label_map.values())
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# --- 5. Additional Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint for health check."""
    return {
        "status": "ok", 
        "message": "Asteroid Risk Prediction API is running.",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/model/info")
def get_model_info():
    """Get information about the loaded model."""
    if not model or not label_map:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {"message": "Metadata file not found"}
    
    return {
        "model_loaded": True,
        "available_classes": list(label_map.values()),
        "training_features": TRAINING_FEATURES,
        "metadata": metadata
    }


@app.get("/model/sample")
def get_sample_data():
    """Get sample input data for testing the API."""
    return {
        "sample_requests": [
            {
                "description": "High-risk asteroid example",
                "data": {
                    "distance_au": 0.018,
                    "velocity_kms": 22.3,
                    "diameter_km": 0.45,
                    "v_infinity_kms": 18.1,
                    "is_pha": True,
                    "orbit_class": "APO"
                }
            },
            {
                "description": "Low-risk asteroid example",
                "data": {
                    "distance_au": 0.15,
                    "velocity_kms": 8.2,
                    "diameter_km": 0.05,
                    "v_infinity_kms": 5.1,
                    "is_pha": False,
                    "orbit_class": "AMO"
                }
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)