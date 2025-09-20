from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np
import json
import os
import asyncio
import logging
from datetime import datetime
from typing import Literal, Dict, Any, Optional, List
import uvicorn

# --- 1. Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Initialize the FastAPI App with Enhanced Configuration ---
app = FastAPI(
    title="StarShield Asteroid Risk Prediction API",
    description="A production-ready API that uses a RandomForest model trained on real NASA JPL CAD data to predict NEO (Near-Earth Object) risk levels with real-time predictions and comprehensive connection capabilities.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# --- 3. Add Middleware for Production Use ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure appropriately for production
)

# --- 4. Global State for Connection Tracking ---
app.state.prediction_count = 0
app.state.start_time = datetime.now()
app.state.connected_clients = set()
app.state.websocket_connections = set()

# --- 5. Load the Trained Model and Labels ---
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


# --- 6. Enhanced Data Models with Validation ---
class AsteroidFeatures(BaseModel):
    object_id: Optional[str] = Field(
        None,
        example="2024-XY47",
        description="NEO object identifier/designation"
    )
    name: Optional[str] = Field(
        None,
        example="2024-XY47", 
        description="NEO object name (often same as object_id)"
    )
    distance_au: float = Field(
        ..., 
        example=0.018, 
        description="Closest approach distance in Astronomical Units.",
        gt=0,
        le=1.0  # Max 1 AU for close approach
    )
    velocity_kms: float = Field(
        ..., 
        example=22.3, 
        description="Relative velocity in km/s.",
        gt=0,
        le=100.0  # Reasonable max velocity
    )
    diameter_km: float = Field(
        ..., 
        example=0.45, 
        description="Estimated diameter in km.",
        gt=0,
        le=100.0  # Max reasonable asteroid size
    )
    v_infinity_kms: float = Field(
        ..., 
        example=18.1, 
        description="Velocity at infinity in km/s.",
        gt=0,
        le=100.0
    )
    is_pha: bool = Field(
        ..., 
        example=True, 
        description="Is the object a Potentially Hazardous Asteroid?"
    )
    orbit_class: Literal['ATE', 'APO', 'AMO', 'IEO'] = Field(
        ..., 
        example='APO', 
        description="The asteroid's orbit class (Apollo, Aten, Amor, Interior Earth Object)."
    )
    eta_closest: Optional[str] = Field(
        None,
        example="2025-09-20T13:45:00Z",
        description="Estimated time of closest approach (ISO format)"
    )
    impact_probability: Optional[float] = Field(
        None,
        example=0.0019,
        description="Estimated impact probability (0-1)",
        ge=0.0,
        le=1.0
    )
    
    @validator('distance_au')
    def validate_distance(cls, v):
        if v > 0.5:
            logger.warning(f"Unusually large approach distance: {v} AU")
        return v
    
    @validator('diameter_km')
    def validate_diameter(cls, v):
        if v > 10.0:
            logger.warning(f"Very large asteroid diameter: {v} km")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "object_id": "2024-XY47",
                "name": "2024-XY47",
                "distance_au": 0.018,
                "velocity_kms": 22.3,
                "diameter_km": 0.45,
                "v_infinity_kms": 18.1,
                "is_pha": True,
                "orbit_class": "APO",
                "eta_closest": "2025-09-20T13:45:00Z",
                "impact_probability": 0.0019
            }
        }

class BatchPredictionRequest(BaseModel):
    asteroids: List[AsteroidFeatures] = Field(
        ...,
        description="List of asteroids to predict",
        min_items=1,
        max_items=100  # Limit batch size
    )


class PredictionResponse(BaseModel):
    # NEO Identification
    object_id: Optional[str] = Field(description="NEO object identifier")
    name: Optional[str] = Field(description="NEO object name")
    
    # Timing and Approach Data
    eta_closest: Optional[str] = Field(description="Estimated time of closest approach")
    distance_km: float = Field(description="Closest approach distance in kilometers")
    distance_au: float = Field(description="Closest approach distance in AU")
    
    # Physical and Motion Properties
    velocity_kms: float = Field(description="Relative velocity in km/s")
    size_km: float = Field(description="Estimated size/diameter in km")
    
    # Risk Assessment
    impact_probability: Optional[float] = Field(description="Estimated impact probability (0-1)")
    predicted_risk_score: float = Field(description="ML model risk score (0-1)")
    predicted_risk_level: str = Field(description="Risk level category")
    confidence: float = Field(description="Prediction confidence")
    
    # Derived Visualization Fields
    time_to_approach_hours: Optional[float] = Field(description="Time to closest approach in hours")
    energy_megaton: float = Field(description="Estimated kinetic energy in megatons")
    
    # Technical Details
    input_features: Dict[str, Any] = Field(description="Input features used for prediction")
    prediction_probabilities: Dict[str, float] = Field(description="Probability distribution across risk levels")
    model_info: Dict[str, Any] = Field(description="Model metadata and information")
    prediction_id: str = Field(description="Unique prediction identifier")
    timestamp: str = Field(description="Prediction timestamp")
    processing_time_ms: float = Field(description="Processing time in milliseconds")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_id: str
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    total_processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float
    total_predictions: int
    current_connections: int


# --- 7. Dependency Injection and Connection Management ---
async def track_connection(request: Request):
    """Track client connections for monitoring."""
    client_ip = request.client.host
    app.state.connected_clients.add(client_ip)
    return client_ip

async def get_model_dependency():
    """Dependency to ensure model is loaded."""
    if not model or not label_map:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server configuration."
        )
    return {"model": model, "label_map": label_map, "features": TRAINING_FEATURES}

# --- Helper functions for calculations ---
def calculate_risk_score(probabilities: np.ndarray) -> float:
    """Calculate a weighted risk score from probabilities."""
    weights = np.array([0, 0.33, 0.66, 1.0])  # Low, Medium, High, Critical
    padded_probs = np.zeros(4)
    padded_probs[:len(probabilities)] = probabilities
    return float(np.dot(padded_probs, weights))


def get_confidence_score(probabilities: np.ndarray) -> float:
    """Calculate confidence as the maximum probability."""
    return float(np.max(probabilities))


def calculate_distance_km(distance_au: float) -> float:
    """Convert AU to kilometers."""
    AU_TO_KM = 149597870.7  # 1 AU in km
    return distance_au * AU_TO_KM


def calculate_kinetic_energy_megaton(diameter_km: float, velocity_kms: float) -> float:
    """
    Calculate estimated kinetic energy in megatons.
    Assumes spherical asteroid with typical density.
    """
    # Typical asteroid density: ~2.5 g/cm³ = 2500 kg/m³
    density_kg_m3 = 2500
    
    # Volume of sphere: (4/3) * π * r³
    radius_m = (diameter_km * 1000) / 2
    volume_m3 = (4/3) * np.pi * (radius_m ** 3)
    
    # Mass in kg
    mass_kg = volume_m3 * density_kg_m3
    
    # Kinetic energy: 0.5 * m * v²
    velocity_ms = velocity_kms * 1000
    kinetic_energy_joules = 0.5 * mass_kg * (velocity_ms ** 2)
    
    # Convert to megatons TNT (1 megaton = 4.184 × 10^15 joules)
    megaton_to_joules = 4.184e15
    kinetic_energy_megaton = kinetic_energy_joules / megaton_to_joules
    
    return float(kinetic_energy_megaton)


def calculate_time_to_approach(eta_closest: Optional[str]) -> Optional[float]:
    """Calculate time to closest approach in hours."""
    if not eta_closest:
        return None
    
    try:
        from datetime import datetime
        approach_time = datetime.fromisoformat(eta_closest.replace('Z', '+00:00'))
        current_time = datetime.now().astimezone()
        time_diff = approach_time - current_time
        return max(0.0, time_diff.total_seconds() / 3600)  # Convert to hours
    except:
        return None


def estimate_impact_probability(distance_au: float, diameter_km: float, is_pha: bool) -> float:
    """
    Estimate impact probability based on distance, size, and PHA status.
    This is a simplified heuristic - real calculations would require orbital mechanics.
    """
    base_probability = 0.0001  # Very low base probability
    
    # Distance factor (closer = higher probability)
    distance_factor = max(0.1, 1.0 / (distance_au * 100))
    
    # Size factor (larger = slightly higher probability of detection/concern)
    size_factor = min(2.0, 1.0 + diameter_km)
    
    # PHA factor
    pha_factor = 3.0 if is_pha else 1.0
    
    estimated_prob = base_probability * distance_factor * size_factor * pha_factor
    return min(0.01, estimated_prob)  # Cap at 1%


# --- 8. Enhanced Prediction Endpoints ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(
    asteroid: AsteroidFeatures,
    background_tasks: BackgroundTasks,
    client_ip: str = Depends(track_connection),
    model_deps: Dict = Depends(get_model_dependency)
):
    """
    Predict the risk level of an asteroid based on its orbital and physical characteristics.
    
    This endpoint uses a RandomForest classifier trained on real NASA JPL Close Approach Data
    to predict risk levels: Low, Medium, High, or Critical.
    
    Features:
    - Real-time prediction with sub-second response times
    - Confidence scoring and probability distributions
    - Connection tracking and monitoring
    - Comprehensive error handling
    """
    import time
    import uuid
    
    start_time = time.time()
    prediction_id = str(uuid.uuid4())
    
    try:
        # Increment prediction counter
        app.state.prediction_count += 1
        
        # Convert input to DataFrame
        input_df_raw = pd.DataFrame([asteroid.model_dump()])
        
        # Apply one-hot encoding for orbit_class (same as training)
        input_df_processed = pd.get_dummies(input_df_raw, columns=['orbit_class'], prefix='class')
        
        # Ensure all training features are present
        input_df_final = input_df_processed.reindex(columns=TRAINING_FEATURES, fill_value=0)
        
        # Make predictions
        prediction_encoded = model_deps["model"].predict(input_df_final)[0]
        probabilities = model_deps["model"].predict_proba(input_df_final)[0]
        predicted_level = model_deps["label_map"][prediction_encoded]
        risk_score = calculate_risk_score(probabilities)
        confidence = get_confidence_score(probabilities)
        
        # Format probability results
        prob_dict = {
            model_deps["label_map"].get(i, f"Unknown_{i}"): round(float(p), 4) 
            for i, p in enumerate(probabilities)
        }
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction, 
            prediction_id, 
            client_ip, 
            predicted_level, 
            confidence,
            processing_time
        )
        
        # Calculate derived visualization fields
        distance_km = calculate_distance_km(asteroid.distance_au)
        energy_megaton = calculate_kinetic_energy_megaton(asteroid.diameter_km, asteroid.velocity_kms)
        time_to_approach = calculate_time_to_approach(asteroid.eta_closest)
        impact_prob = asteroid.impact_probability or estimate_impact_probability(
            asteroid.distance_au, asteroid.diameter_km, asteroid.is_pha
        )
        
        return PredictionResponse(
            # NEO Identification
            object_id=asteroid.object_id or f"NEO-{prediction_id[:8]}",
            name=asteroid.name or asteroid.object_id or f"NEO-{prediction_id[:8]}",
            
            # Timing and Approach Data
            eta_closest=asteroid.eta_closest,
            distance_km=round(distance_km, 2),
            distance_au=round(asteroid.distance_au, 6),
            
            # Physical and Motion Properties
            velocity_kms=round(asteroid.velocity_kms, 2),
            size_km=round(asteroid.diameter_km, 3),
            
            # Risk Assessment
            impact_probability=round(impact_prob, 6),
            predicted_risk_score=round(risk_score, 4),
            predicted_risk_level=predicted_level,
            confidence=round(confidence, 4),
            
            # Derived Visualization Fields
            time_to_approach_hours=round(time_to_approach, 2) if time_to_approach else None,
            energy_megaton=round(energy_megaton, 2),
            
            # Technical Details
            input_features=asteroid.model_dump(),
            prediction_probabilities=prob_dict,
            model_info={
                "model_type": "RandomForestClassifier",
                "features_used": len(TRAINING_FEATURES),
                "available_classes": list(model_deps["label_map"].values()),
                "training_data_source": "NASA JPL CAD API"
            },
            prediction_id=prediction_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Prediction failed for {prediction_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    client_ip: str = Depends(track_connection),
    model_deps: Dict = Depends(get_model_dependency)
):
    """
    Predict risk levels for multiple asteroids in a single request.
    
    Supports batch processing up to 100 asteroids at once for efficient
    bulk predictions with comprehensive error handling per item.
    """
    import time
    import uuid
    
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    predictions = []
    successful = 0
    failed = 0
    
    logger.info(f"Processing batch {batch_id} with {len(batch_request.asteroids)} asteroids from {client_ip}")
    
    for i, asteroid in enumerate(batch_request.asteroids):
        try:
            # Process individual prediction
            input_df_raw = pd.DataFrame([asteroid.model_dump()])
            input_df_processed = pd.get_dummies(input_df_raw, columns=['orbit_class'], prefix='class')
            input_df_final = input_df_processed.reindex(columns=TRAINING_FEATURES, fill_value=0)
            
            prediction_encoded = model_deps["model"].predict(input_df_final)[0]
            probabilities = model_deps["model"].predict_proba(input_df_final)[0]
            predicted_level = model_deps["label_map"][prediction_encoded]
            risk_score = calculate_risk_score(probabilities)
            confidence = get_confidence_score(probabilities)
            
            prob_dict = {
                model_deps["label_map"].get(j, f"Unknown_{j}"): round(float(p), 4) 
                for j, p in enumerate(probabilities)
            }
            
            prediction_id = f"{batch_id}-{i}"
            item_processing_time = (time.time() - start_time) * 1000
            
            # Calculate derived visualization fields
            distance_km = calculate_distance_km(asteroid.distance_au)
            energy_megaton = calculate_kinetic_energy_megaton(asteroid.diameter_km, asteroid.velocity_kms)
            time_to_approach = calculate_time_to_approach(asteroid.eta_closest)
            impact_prob = asteroid.impact_probability or estimate_impact_probability(
                asteroid.distance_au, asteroid.diameter_km, asteroid.is_pha
            )
            
            predictions.append(PredictionResponse(
                # NEO Identification
                object_id=asteroid.object_id or f"NEO-{prediction_id}",
                name=asteroid.name or asteroid.object_id or f"NEO-{prediction_id}",
                
                # Timing and Approach Data
                eta_closest=asteroid.eta_closest,
                distance_km=round(distance_km, 2),
                distance_au=round(asteroid.distance_au, 6),
                
                # Physical and Motion Properties
                velocity_kms=round(asteroid.velocity_kms, 2),
                size_km=round(asteroid.diameter_km, 3),
                
                # Risk Assessment
                impact_probability=round(impact_prob, 6),
                predicted_risk_score=round(risk_score, 4),
                predicted_risk_level=predicted_level,
                confidence=round(confidence, 4),
                
                # Derived Visualization Fields
                time_to_approach_hours=round(time_to_approach, 2) if time_to_approach else None,
                energy_megaton=round(energy_megaton, 2),
                
                # Technical Details
                input_features=asteroid.model_dump(),
                prediction_probabilities=prob_dict,
                model_info={
                    "model_type": "RandomForestClassifier",
                    "features_used": len(TRAINING_FEATURES),
                    "available_classes": list(model_deps["label_map"].values())
                },
                prediction_id=prediction_id,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=round(item_processing_time, 2)
            ))
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed to process asteroid {i} in batch {batch_id}: {str(e)}")
            failed += 1
    
    total_processing_time = (time.time() - start_time) * 1000
    app.state.prediction_count += successful
    
    # Log batch processing
    background_tasks.add_task(
        log_batch_prediction,
        batch_id,
        client_ip,
        len(batch_request.asteroids),
        successful,
        failed,
        total_processing_time
    )
    
    return BatchPredictionResponse(
        predictions=predictions,
        batch_id=batch_id,
        total_predictions=len(batch_request.asteroids),
        successful_predictions=successful,
        failed_predictions=failed,
        total_processing_time_ms=round(total_processing_time, 2)
    )

# Background task functions
async def log_prediction(prediction_id: str, client_ip: str, risk_level: str, confidence: float, processing_time: float):
    """Log prediction details for monitoring and analytics."""
    logger.info(f"Prediction {prediction_id} from {client_ip}: {risk_level} (confidence: {confidence:.2f}, time: {processing_time:.2f}ms)")

async def log_batch_prediction(batch_id: str, client_ip: str, total: int, successful: int, failed: int, processing_time: float):
    """Log batch prediction details."""
    logger.info(f"Batch {batch_id} from {client_ip}: {successful}/{total} successful, {failed} failed, time: {processing_time:.2f}ms")


# --- 8.5. Data Endpoints for Training and Test Sets ---

@app.get("/data/train", response_model=BatchPredictionResponse)
def get_train_predictions(
    background_tasks: BackgroundTasks, 
    client_ip: str = Depends(track_connection),
    model_deps: Dict = Depends(get_model_dependency)
):
    """
    Get predictions for all asteroids in the training dataset.
    """
    import time
    batch_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Load training data
        with open('real_asteroid_data_train.json', 'r') as f:
            train_data = json.load(f)
        
        # Convert to batch prediction format
        asteroids = []
        for record in train_data:
            asteroids.append(AsteroidFeatures(
                object_id=record.get('object_name'),
                name=record.get('object_name'),
                distance_au=record['distance_au'],
                velocity_kms=record['velocity_kms'],
                diameter_km=record['diameter_km'],
                v_infinity_kms=record['v_infinity_kms'],
                is_pha=record['is_pha'],
                orbit_class=record['orbit_class'],
                eta_closest=record.get('approach_date'),
                impact_probability=None  # Will be calculated
            ))
        
        # Process predictions using the same logic as the batch endpoint
        start_time = time.time()
        predictions = []
        successful = 0
        failed = 0
        
        for i, asteroid in enumerate(asteroids):
            try:
                item_start_time = time.time()
                
                # Convert input to DataFrame (same as single predict)
                input_df_raw = pd.DataFrame([asteroid.model_dump()])
                
                # Apply one-hot encoding for orbit_class (same as training)
                input_df_processed = pd.get_dummies(input_df_raw, columns=['orbit_class'], prefix='class')
                
                # Ensure all training features are present
                input_df_final = input_df_processed.reindex(columns=TRAINING_FEATURES, fill_value=0)
                
                # Make predictions
                prediction_encoded = model_deps["model"].predict(input_df_final)[0]
                probabilities = model_deps["model"].predict_proba(input_df_final)[0]
                predicted_level = model_deps["label_map"][prediction_encoded]
                risk_score = calculate_risk_score(probabilities)
                confidence = get_confidence_score(probabilities)
                
                # Format probability results
                prob_dict = {
                    model_deps["label_map"].get(j, f"Unknown_{j}"): round(float(p), 4) 
                    for j, p in enumerate(probabilities)
                }
                
                item_processing_time = (time.time() - item_start_time) * 1000
                prediction_id = f"{batch_id}_item_{i}"
                
                # Calculate derived visualization fields
                distance_km = calculate_distance_km(asteroid.distance_au)
                energy_megaton = calculate_kinetic_energy_megaton(asteroid.diameter_km, asteroid.velocity_kms)
                time_to_approach = calculate_time_to_approach(asteroid.eta_closest)
                impact_prob = asteroid.impact_probability or estimate_impact_probability(
                    asteroid.distance_au, asteroid.diameter_km, asteroid.is_pha
                )
                
                predictions.append(PredictionResponse(
                    # NEO Identification
                    object_id=asteroid.object_id or f"TRAIN-{prediction_id}",
                    name=asteroid.name or asteroid.object_id or f"TRAIN-{prediction_id}",
                    
                    # Timing and Approach Data
                    eta_closest=asteroid.eta_closest,
                    distance_km=round(distance_km, 2),
                    distance_au=round(asteroid.distance_au, 6),
                    
                    # Physical and Motion Properties
                    velocity_kms=round(asteroid.velocity_kms, 2),
                    size_km=round(asteroid.diameter_km, 3),
                    
                    # Risk Assessment
                    impact_probability=round(impact_prob, 6),
                    predicted_risk_score=round(risk_score, 4),
                    predicted_risk_level=predicted_level,
                    confidence=round(confidence, 4),
                    
                    # Derived Visualization Fields
                    time_to_approach_hours=round(time_to_approach, 2) if time_to_approach else None,
                    energy_megaton=round(energy_megaton, 2),
                    
                    # Technical Details
                    input_features=asteroid.model_dump(),
                    prediction_probabilities=prob_dict,
                    model_info={
                        "model_type": "RandomForestClassifier",
                        "features_used": len(TRAINING_FEATURES),
                        "available_classes": list(model_deps["label_map"].values())
                    },
                    prediction_id=prediction_id,
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=round(item_processing_time, 2)
                ))
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to process training record {i}: {str(e)}")
                failed += 1
        
        total_processing_time = (time.time() - start_time) * 1000
        app.state.prediction_count += successful
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            total_predictions=len(asteroids),
            successful_predictions=successful,
            failed_predictions=failed,
            total_processing_time_ms=round(total_processing_time, 2)
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Training data file not found. Run convert_cad_data.py first.")
    except Exception as e:
        logger.error(f"Error processing training data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing training data: {str(e)}")


@app.get("/data/test", response_model=BatchPredictionResponse)
def get_test_predictions(
    background_tasks: BackgroundTasks, 
    client_ip: str = Depends(track_connection),
    model_deps: Dict = Depends(get_model_dependency)
):
    """
    Get predictions for all asteroids in the test dataset.
    """
    import time
    batch_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Load test data
        with open('real_asteroid_data_test.json', 'r') as f:
            test_data = json.load(f)
        
        # Convert to batch prediction format
        asteroids = []
        for record in test_data:
            asteroids.append(AsteroidFeatures(
                object_id=record.get('object_name'),
                name=record.get('object_name'),
                distance_au=record['distance_au'],
                velocity_kms=record['velocity_kms'],
                diameter_km=record['diameter_km'],
                v_infinity_kms=record['v_infinity_kms'],
                is_pha=record['is_pha'],
                orbit_class=record['orbit_class'],
                eta_closest=record.get('approach_date'),
                impact_probability=None  # Will be calculated
            ))
        
        # Process predictions using the same logic as the batch endpoint
        start_time = time.time()
        predictions = []
        successful = 0
        failed = 0
        
        for i, asteroid in enumerate(asteroids):
            try:
                item_start_time = time.time()
                
                # Convert input to DataFrame (same as single predict)
                input_df_raw = pd.DataFrame([asteroid.model_dump()])
                
                # Apply one-hot encoding for orbit_class (same as training)
                input_df_processed = pd.get_dummies(input_df_raw, columns=['orbit_class'], prefix='class')
                
                # Ensure all training features are present
                input_df_final = input_df_processed.reindex(columns=TRAINING_FEATURES, fill_value=0)
                
                # Make predictions
                prediction_encoded = model_deps["model"].predict(input_df_final)[0]
                probabilities = model_deps["model"].predict_proba(input_df_final)[0]
                predicted_level = model_deps["label_map"][prediction_encoded]
                risk_score = calculate_risk_score(probabilities)
                confidence = get_confidence_score(probabilities)
                
                # Format probability results
                prob_dict = {
                    model_deps["label_map"].get(j, f"Unknown_{j}"): round(float(p), 4) 
                    for j, p in enumerate(probabilities)
                }
                
                item_processing_time = (time.time() - item_start_time) * 1000
                prediction_id = f"{batch_id}_item_{i}"
                
                # Calculate derived visualization fields
                distance_km = calculate_distance_km(asteroid.distance_au)
                energy_megaton = calculate_kinetic_energy_megaton(asteroid.diameter_km, asteroid.velocity_kms)
                time_to_approach = calculate_time_to_approach(asteroid.eta_closest)
                impact_prob = asteroid.impact_probability or estimate_impact_probability(
                    asteroid.distance_au, asteroid.diameter_km, asteroid.is_pha
                )
                
                predictions.append(PredictionResponse(
                    # NEO Identification
                    object_id=asteroid.object_id or f"TEST-{prediction_id}",
                    name=asteroid.name or asteroid.object_id or f"TEST-{prediction_id}",
                    
                    # Timing and Approach Data
                    eta_closest=asteroid.eta_closest,
                    distance_km=round(distance_km, 2),
                    distance_au=round(asteroid.distance_au, 6),
                    
                    # Physical and Motion Properties
                    velocity_kms=round(asteroid.velocity_kms, 2),
                    size_km=round(asteroid.diameter_km, 3),
                    
                    # Risk Assessment
                    impact_probability=round(impact_prob, 6),
                    predicted_risk_score=round(risk_score, 4),
                    predicted_risk_level=predicted_level,
                    confidence=round(confidence, 4),
                    
                    # Derived Visualization Fields
                    time_to_approach_hours=round(time_to_approach, 2) if time_to_approach else None,
                    energy_megaton=round(energy_megaton, 2),
                    
                    # Technical Details
                    input_features=asteroid.model_dump(),
                    prediction_probabilities=prob_dict,
                    model_info={
                        "model_type": "RandomForestClassifier",
                        "features_used": len(TRAINING_FEATURES),
                        "available_classes": list(model_deps["label_map"].values())
                    },
                    prediction_id=prediction_id,
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=round(item_processing_time, 2)
                ))
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to process test record {i}: {str(e)}")
                failed += 1
        
        total_processing_time = (time.time() - start_time) * 1000
        app.state.prediction_count += successful
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            total_predictions=len(asteroids),
            successful_predictions=successful,
            failed_predictions=failed,
            total_processing_time_ms=round(total_processing_time, 2)
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test data file not found. Run convert_cad_data.py first.")
    except Exception as e:
        logger.error(f"Error processing test data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing test data: {str(e)}")


# --- 8.7. Individual Record Endpoints ---

@app.get("/data/train/{index}", response_model=PredictionResponse)
def get_train_record(
    index: int,
    background_tasks: BackgroundTasks,
    client_ip: str = Depends(track_connection),
    model_deps: Dict = Depends(get_model_dependency)
):
    """
    Get prediction for a specific asteroid from the training dataset by index.
    
    Args:
        index: Zero-based index of the record in the training dataset
    """
    import time
    
    try:
        # Load training data
        with open('real_asteroid_data_train.json', 'r') as f:
            train_data = json.load(f)
        
        # Validate index
        if index < 0 or index >= len(train_data):
            raise HTTPException(
                status_code=404, 
                detail=f"Training record index {index} not found. Valid range: 0-{len(train_data)-1}"
            )
        
        # Get the specific record
        record = train_data[index]
        asteroid = AsteroidFeatures(
            object_id=record.get('object_name'),
            name=record.get('object_name'),
            distance_au=record['distance_au'],
            velocity_kms=record['velocity_kms'],
            diameter_km=record['diameter_km'],
            v_infinity_kms=record['v_infinity_kms'],
            is_pha=record['is_pha'],
            orbit_class=record['orbit_class'],
            eta_closest=record.get('approach_date'),
            impact_probability=None  # Will be calculated
        )
        
        # Process prediction
        start_time = time.time()
        prediction_id = f"train_record_{index}"
        
        # Convert input to DataFrame
        input_df_raw = pd.DataFrame([asteroid.model_dump()])
        input_df_processed = pd.get_dummies(input_df_raw, columns=['orbit_class'], prefix='class')
        input_df_final = input_df_processed.reindex(columns=TRAINING_FEATURES, fill_value=0)
        
        # Make predictions
        prediction_encoded = model_deps["model"].predict(input_df_final)[0]
        probabilities = model_deps["model"].predict_proba(input_df_final)[0]
        predicted_level = model_deps["label_map"][prediction_encoded]
        risk_score = calculate_risk_score(probabilities)
        confidence = get_confidence_score(probabilities)
        
        # Format probability results
        prob_dict = {
            model_deps["label_map"].get(j, f"Unknown_{j}"): round(float(p), 4) 
            for j, p in enumerate(probabilities)
        }
        
        # Calculate derived visualization fields
        distance_km = calculate_distance_km(asteroid.distance_au)
        energy_megaton = calculate_kinetic_energy_megaton(asteroid.diameter_km, asteroid.velocity_kms)
        time_to_approach = calculate_time_to_approach(asteroid.eta_closest)
        impact_prob = asteroid.impact_probability or estimate_impact_probability(
            asteroid.distance_au, asteroid.diameter_km, asteroid.is_pha
        )
        
        processing_time = (time.time() - start_time) * 1000
        app.state.prediction_count += 1
        
        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction, 
            prediction_id, 
            client_ip, 
            predicted_level, 
            confidence,
            processing_time
        )
        
        return PredictionResponse(
            # NEO Identification
            object_id=asteroid.object_id or f"TRAIN-{index}",
            name=asteroid.name or asteroid.object_id or f"TRAIN-{index}",
            
            # Timing and Approach Data
            eta_closest=asteroid.eta_closest,
            distance_km=round(distance_km, 2),
            distance_au=round(asteroid.distance_au, 6),
            
            # Physical and Motion Properties
            velocity_kms=round(asteroid.velocity_kms, 2),
            size_km=round(asteroid.diameter_km, 3),
            
            # Risk Assessment
            impact_probability=round(impact_prob, 6),
            predicted_risk_score=round(risk_score, 4),
            predicted_risk_level=predicted_level,
            confidence=round(confidence, 4),
            
            # Derived Visualization Fields
            time_to_approach_hours=round(time_to_approach, 2) if time_to_approach else None,
            energy_megaton=round(energy_megaton, 2),
            
            # Technical Details
            input_features=asteroid.model_dump(),
            prediction_probabilities=prob_dict,
            model_info={
                "model_type": "RandomForestClassifier",
                "features_used": len(TRAINING_FEATURES),
                "available_classes": list(model_deps["label_map"].values()),
                "record_index": index,
                "dataset": "training",
                "total_records": len(train_data)
            },
            prediction_id=prediction_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2)
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Training data file not found. Run convert_cad_data.py first.")
    except Exception as e:
        logger.error(f"Error processing training record {index}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing training record: {str(e)}")


@app.get("/data/test/{index}", response_model=PredictionResponse)
def get_test_record(
    index: int,
    background_tasks: BackgroundTasks,
    client_ip: str = Depends(track_connection),
    model_deps: Dict = Depends(get_model_dependency)
):
    """
    Get prediction for a specific asteroid from the test dataset by index.
    
    Args:
        index: Zero-based index of the record in the test dataset
    """
    import time
    
    try:
        # Load test data
        with open('real_asteroid_data_test.json', 'r') as f:
            test_data = json.load(f)
        
        # Validate index
        if index < 0 or index >= len(test_data):
            raise HTTPException(
                status_code=404, 
                detail=f"Test record index {index} not found. Valid range: 0-{len(test_data)-1}"
            )
        
        # Get the specific record
        record = test_data[index]
        asteroid = AsteroidFeatures(
            object_id=record.get('object_name'),
            name=record.get('object_name'),
            distance_au=record['distance_au'],
            velocity_kms=record['velocity_kms'],
            diameter_km=record['diameter_km'],
            v_infinity_kms=record['v_infinity_kms'],
            is_pha=record['is_pha'],
            orbit_class=record['orbit_class'],
            eta_closest=record.get('approach_date'),
            impact_probability=None  # Will be calculated
        )
        
        # Process prediction
        start_time = time.time()
        prediction_id = f"test_record_{index}"
        
        # Convert input to DataFrame
        input_df_raw = pd.DataFrame([asteroid.model_dump()])
        input_df_processed = pd.get_dummies(input_df_raw, columns=['orbit_class'], prefix='class')
        input_df_final = input_df_processed.reindex(columns=TRAINING_FEATURES, fill_value=0)
        
        # Make predictions
        prediction_encoded = model_deps["model"].predict(input_df_final)[0]
        probabilities = model_deps["model"].predict_proba(input_df_final)[0]
        predicted_level = model_deps["label_map"][prediction_encoded]
        risk_score = calculate_risk_score(probabilities)
        confidence = get_confidence_score(probabilities)
        
        # Format probability results
        prob_dict = {
            model_deps["label_map"].get(j, f"Unknown_{j}"): round(float(p), 4) 
            for j, p in enumerate(probabilities)
        }
        
        # Calculate derived visualization fields
        distance_km = calculate_distance_km(asteroid.distance_au)
        energy_megaton = calculate_kinetic_energy_megaton(asteroid.diameter_km, asteroid.velocity_kms)
        time_to_approach = calculate_time_to_approach(asteroid.eta_closest)
        impact_prob = asteroid.impact_probability or estimate_impact_probability(
            asteroid.distance_au, asteroid.diameter_km, asteroid.is_pha
        )
        
        processing_time = (time.time() - start_time) * 1000
        app.state.prediction_count += 1
        
        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction, 
            prediction_id, 
            client_ip, 
            predicted_level, 
            confidence,
            processing_time
        )
        
        return PredictionResponse(
            # NEO Identification
            object_id=asteroid.object_id or f"TEST-{index}",
            name=asteroid.name or asteroid.object_id or f"TEST-{index}",
            
            # Timing and Approach Data
            eta_closest=asteroid.eta_closest,
            distance_km=round(distance_km, 2),
            distance_au=round(asteroid.distance_au, 6),
            
            # Physical and Motion Properties
            velocity_kms=round(asteroid.velocity_kms, 2),
            size_km=round(asteroid.diameter_km, 3),
            
            # Risk Assessment
            impact_probability=round(impact_prob, 6),
            predicted_risk_score=round(risk_score, 4),
            predicted_risk_level=predicted_level,
            confidence=round(confidence, 4),
            
            # Derived Visualization Fields
            time_to_approach_hours=round(time_to_approach, 2) if time_to_approach else None,
            energy_megaton=round(energy_megaton, 2),
            
            # Technical Details
            input_features=asteroid.model_dump(),
            prediction_probabilities=prob_dict,
            model_info={
                "model_type": "RandomForestClassifier",
                "features_used": len(TRAINING_FEATURES),
                "available_classes": list(model_deps["label_map"].values()),
                "record_index": index,
                "dataset": "test",
                "total_records": len(test_data)
            },
            prediction_id=prediction_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2)
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test data file not found. Run convert_cad_data.py first.")
    except Exception as e:
        logger.error(f"Error processing test record {index}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing test record: {str(e)}")


@app.get("/data/all", response_model=BatchPredictionResponse)
def get_all_predictions(
    background_tasks: BackgroundTasks, 
    client_ip: str = Depends(track_connection),
    model_deps: Dict = Depends(get_model_dependency)
):
    """
    Get predictions for all asteroids from both training and test datasets combined.
    Returns the complete dataset (1300 records) with predictions.
    """
    import time
    batch_id = f"all_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Load both training and test data
        with open('real_asteroid_data_train.json', 'r') as f:
            train_data = json.load(f)
        
        with open('real_asteroid_data_test.json', 'r') as f:
            test_data = json.load(f)
        
        # Combine datasets
        all_data = []
        
        # Add training data with metadata
        for i, record in enumerate(train_data):
            record_with_meta = record.copy()
            record_with_meta['dataset_source'] = 'training'
            record_with_meta['original_index'] = i
            all_data.append(record_with_meta)
        
        # Add test data with metadata
        for i, record in enumerate(test_data):
            record_with_meta = record.copy()
            record_with_meta['dataset_source'] = 'test'
            record_with_meta['original_index'] = i
            all_data.append(record_with_meta)
        
        # Convert to batch prediction format
        asteroids = []
        for record in all_data:
            asteroids.append(AsteroidFeatures(
                object_id=record.get('object_name'),
                name=record.get('object_name'),
                distance_au=record['distance_au'],
                velocity_kms=record['velocity_kms'],
                diameter_km=record['diameter_km'],
                v_infinity_kms=record['v_infinity_kms'],
                is_pha=record['is_pha'],
                orbit_class=record['orbit_class'],
                eta_closest=record.get('approach_date'),
                impact_probability=None  # Will be calculated
            ))
        
        # Process predictions using the same logic as the batch endpoint
        start_time = time.time()
        predictions = []
        successful = 0
        failed = 0
        
        for i, (asteroid, record) in enumerate(zip(asteroids, all_data)):
            try:
                item_start_time = time.time()
                
                # Convert input to DataFrame (same as single predict)
                input_df_raw = pd.DataFrame([asteroid.model_dump()])
                
                # Apply one-hot encoding for orbit_class (same as training)
                input_df_processed = pd.get_dummies(input_df_raw, columns=['orbit_class'], prefix='class')
                
                # Ensure all training features are present
                input_df_final = input_df_processed.reindex(columns=TRAINING_FEATURES, fill_value=0)
                
                # Make predictions
                prediction_encoded = model_deps["model"].predict(input_df_final)[0]
                probabilities = model_deps["model"].predict_proba(input_df_final)[0]
                predicted_level = model_deps["label_map"][prediction_encoded]
                risk_score = calculate_risk_score(probabilities)
                confidence = get_confidence_score(probabilities)
                
                # Format probability results
                prob_dict = {
                    model_deps["label_map"].get(j, f"Unknown_{j}"): round(float(p), 4) 
                    for j, p in enumerate(probabilities)
                }
                
                item_processing_time = (time.time() - item_start_time) * 1000
                prediction_id = f"{batch_id}_item_{i}"
                
                # Calculate derived visualization fields
                distance_km = calculate_distance_km(asteroid.distance_au)
                energy_megaton = calculate_kinetic_energy_megaton(asteroid.diameter_km, asteroid.velocity_kms)
                time_to_approach = calculate_time_to_approach(asteroid.eta_closest)
                impact_prob = asteroid.impact_probability or estimate_impact_probability(
                    asteroid.distance_au, asteroid.diameter_km, asteroid.is_pha
                )
                
                predictions.append(PredictionResponse(
                    # NEO Identification
                    object_id=asteroid.object_id or f"ALL-{prediction_id}",
                    name=asteroid.name or asteroid.object_id or f"ALL-{prediction_id}",
                    
                    # Timing and Approach Data
                    eta_closest=asteroid.eta_closest,
                    distance_km=round(distance_km, 2),
                    distance_au=round(asteroid.distance_au, 6),
                    
                    # Physical and Motion Properties
                    velocity_kms=round(asteroid.velocity_kms, 2),
                    size_km=round(asteroid.diameter_km, 3),
                    
                    # Risk Assessment
                    impact_probability=round(impact_prob, 6),
                    predicted_risk_score=round(risk_score, 4),
                    predicted_risk_level=predicted_level,
                    confidence=round(confidence, 4),
                    
                    # Derived Visualization Fields
                    time_to_approach_hours=round(time_to_approach, 2) if time_to_approach else None,
                    energy_megaton=round(energy_megaton, 2),
                    
                    # Technical Details
                    input_features=asteroid.model_dump(),
                    prediction_probabilities=prob_dict,
                    model_info={
                        "model_type": "RandomForestClassifier",
                        "features_used": len(TRAINING_FEATURES),
                        "available_classes": list(model_deps["label_map"].values()),
                        "dataset_source": record['dataset_source'],
                        "original_index": record['original_index'],
                        "combined_index": i
                    },
                    prediction_id=prediction_id,
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=round(item_processing_time, 2)
                ))
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to process combined record {i}: {str(e)}")
                failed += 1
        
        total_processing_time = (time.time() - start_time) * 1000
        app.state.prediction_count += successful
        
        # Log batch processing
        background_tasks.add_task(
            log_batch_prediction,
            batch_id,
            client_ip,
            len(asteroids),
            successful,
            failed,
            total_processing_time
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            total_predictions=len(asteroids),
            successful_predictions=successful,
            failed_predictions=failed,
            total_processing_time_ms=round(total_processing_time, 2)
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data files not found: {str(e)}. Run convert_cad_data.py first.")
    except Exception as e:
        logger.error(f"Error processing combined data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing combined data: {str(e)}")


# --- 9. Enhanced Health and Monitoring Endpoints ---
@app.get("/", response_model=HealthResponse)
async def read_root():
    """
    Root endpoint providing comprehensive health check and system status.
    
    Returns current system status, model state, uptime, and connection metrics.
    """
    uptime = (datetime.now() - app.state.start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if model and label_map else "degraded",
        model_loaded=model is not None and label_map is not None,
        version="2.0.0",
        uptime_seconds=round(uptime, 2),
        total_predictions=app.state.prediction_count,
        current_connections=len(app.state.connected_clients)
    )

@app.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe endpoint."""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}

@app.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe endpoint."""
    if not model or not label_map:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready", "model_loaded": True, "timestamp": datetime.now().isoformat()}


@app.get("/model/info")
async def get_model_info(model_deps: Dict = Depends(get_model_dependency)):
    """
    Get comprehensive information about the loaded model.
    
    Includes training metadata, feature importance, model parameters,
    and performance metrics.
    """
    try:
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {"message": "Metadata file not found"}
    
    # Add runtime information
    model_info = {
        "model_loaded": True,
        "available_classes": list(model_deps["label_map"].values()),
        "training_features": TRAINING_FEATURES,
        "feature_count": len(TRAINING_FEATURES),
        "metadata": metadata,
        "runtime_stats": {
            "total_predictions": app.state.prediction_count,
            "uptime_seconds": (datetime.now() - app.state.start_time).total_seconds(),
            "connected_clients": len(app.state.connected_clients)
        }
    }
    
    # Add feature importance if available
    try:
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(TRAINING_FEATURES, model.feature_importances_))
            model_info["feature_importance"] = {
                k: round(float(v), 4) for k, v in feature_importance.items()
            }
    except Exception as e:
        logger.warning(f"Could not get feature importance: {e}")
    
    return model_info

@app.get("/metrics")
async def get_metrics():
    """
    Get system metrics for monitoring and observability.
    
    Provides detailed metrics for monitoring tools like Prometheus.
    """
    uptime = (datetime.now() - app.state.start_time).total_seconds()
    
    return {
        "system": {
            "uptime_seconds": round(uptime, 2),
            "start_time": app.state.start_time.isoformat(),
            "current_time": datetime.now().isoformat()
        },
        "api": {
            "total_predictions": app.state.prediction_count,
            "predictions_per_second": round(app.state.prediction_count / max(uptime, 1), 4),
            "connected_clients": len(app.state.connected_clients),
            "active_connections": len(app.state.connected_clients)
        },
        "model": {
            "loaded": model is not None and label_map is not None,
            "features_count": len(TRAINING_FEATURES) if TRAINING_FEATURES else 0,
            "classes_count": len(label_map) if label_map else 0
        }
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


# --- 10. WebSocket Support for Real-time Connections ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time asteroid risk predictions.
    
    Supports streaming predictions and real-time monitoring.
    """
    await websocket.accept()
    app.state.websocket_connections.add(websocket)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            if data.get("type") == "predict":
                try:
                    # Validate asteroid data
                    asteroid_data = data.get("asteroid")
                    asteroid = AsteroidFeatures(**asteroid_data)
                    
                    # Make prediction
                    input_df_raw = pd.DataFrame([asteroid.model_dump()])
                    input_df_processed = pd.get_dummies(input_df_raw, columns=['orbit_class'], prefix='class')
                    input_df_final = input_df_processed.reindex(columns=TRAINING_FEATURES, fill_value=0)
                    
                    prediction_encoded = model.predict(input_df_final)[0]
                    probabilities = model.predict_proba(input_df_final)[0]
                    predicted_level = label_map[prediction_encoded]
                    risk_score = calculate_risk_score(probabilities)
                    confidence = get_confidence_score(probabilities)
                    
                    prob_dict = {
                        label_map.get(i, f"Unknown_{i}"): round(float(p), 4) 
                        for i, p in enumerate(probabilities)
                    }
                    
                    # Send prediction result
                    await websocket.send_json({
                        "type": "prediction_result",
                        "prediction": {
                            "predicted_risk_level": predicted_level,
                            "predicted_risk_score": round(risk_score, 4),
                            "confidence": round(confidence, 4),
                            "prediction_probabilities": prob_dict,
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                    
                    app.state.prediction_count += 1
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Prediction failed: {str(e)}"
                    })
                    
            elif data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        app.state.websocket_connections.discard(websocket)

# --- 11. Application Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("🚀 StarShield Asteroid Risk Prediction API starting up...")
    logger.info(f"📊 Model loaded: {model is not None}")
    logger.info(f"🏷️  Label map loaded: {label_map is not None}")
    logger.info(f"🔧 Features available: {len(TRAINING_FEATURES) if TRAINING_FEATURES else 0}")
    
    if model and label_map:
        try:
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
            logger.info(f"📈 Training records: {metadata.get('training_records', 'unknown')}")
            logger.info(f"🎯 Model type: {metadata.get('model_type', 'unknown')}")
        except FileNotFoundError:
            logger.warning("⚠️  Model metadata not found")
    
    logger.info("✅ Application startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("🛑 StarShield API shutting down...")
    
    # Close all WebSocket connections
    for websocket in app.state.websocket_connections.copy():
        try:
            await websocket.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket: {e}")
    
    logger.info(f"📊 Final statistics:")
    logger.info(f"   Total predictions: {app.state.prediction_count}")
    logger.info(f"   Connected clients: {len(app.state.connected_clients)}")
    uptime = (datetime.now() - app.state.start_time).total_seconds()
    logger.info(f"   Uptime: {uptime:.2f} seconds")
    logger.info("👋 Shutdown complete!")

# --- 12. Main Application Entry Point ---
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🌟 Starting StarShield Asteroid Risk Prediction API")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        log_level="info",
        access_log=True
    )