# Asteroid Risk Prediction API (JSON Version)

This is an updated version of the Asteroid Risk Prediction API that uses JSON files instead of CSV files for training data.

## Files Overview

- `train_model.py` - Training script that loads JSON data and trains the model
- `asteroid_api.py` - FastAPI application for serving predictions
- `sample_asteroid_data.json` - Example JSON data file showing the expected format

## Setup Instructions

### 1. Install Requirements

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib matplotlib seaborn pydantic
```

### 2. Prepare Your Training Data

Create a file named `real_asteroid_data.json` with your training data. The JSON file should contain an array of objects with the following structure:

```json
[
  {
    "distance_au": 0.018,
    "velocity_kms": 22.3,
    "diameter_km": 0.45,
    "v_infinity_kms": 18.1,
    "is_pha": true,
    "orbit_class": "APO",
    "risk_level": "High"
  }
]
```

**Required Fields:**
- `distance_au` (float): Closest approach distance in Astronomical Units
- `velocity_kms` (float): Relative velocity in km/s
- `diameter_km` (float): Estimated diameter in km
- `v_infinity_kms` (float): Velocity at infinity in km/s
- `is_pha` (boolean): Is the object a Potentially Hazardous Asteroid?
- `orbit_class` (string): Must be one of: "ATE", "APO", "AMO", "IEO"
- `risk_level` (string): Must be one of: "Low", "Medium", "High", "Critical"

### 3. Train the Model

```bash
python train_model.py
```

This will:
- Load your `real_asteroid_data.json` file
- Train a RandomForest model
- Save the model files: `asteroid_risk_model.joblib` and `risk_level_labels.joblib`
- Save training metadata: `model_metadata.json`
- Display training results and feature importance

### 4. Start the API

```bash
python asteroid_api.py
```

Or using uvicorn directly:
```bash
uvicorn asteroid_api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: http://localhost:8000

## API Endpoints

### POST /predict
Make a risk prediction for an asteroid.

**Request Body:**
```json
{
  "distance_au": 0.018,
  "velocity_kms": 22.3,
  "diameter_km": 0.45,
  "v_infinity_kms": 18.1,
  "is_pha": true,
  "orbit_class": "APO"
}
```

**Response:**
```json
{
  "input_features": { ... },
  "predicted_risk_level": "High",
  "predicted_risk_score": 0.7523,
  "confidence": 0.8456,
  "prediction_probabilities": {
    "Low": 0.0234,
    "Medium": 0.1543,
    "High": 0.8456,
    "Critical": 0.0767
  },
  "model_info": {
    "model_type": "RandomForestClassifier",
    "features_used": 9,
    "available_classes": ["Low", "Medium", "High", "Critical"]
  }
}
```

### GET /
Health check endpoint.

### GET /model/info
Get information about the loaded model.

### GET /model/sample
Get sample input data for testing.

## Key Improvements in JSON Version

1. **Better Error Handling**: More detailed error messages for data validation
2. **Metadata Tracking**: Saves training metadata for better model management
3. **Enhanced API**: More comprehensive response format with confidence scores
4. **Data Validation**: Better validation of input data format and required fields
5. **Flexible Format**: JSON is more flexible than CSV for complex data structures

## Testing with Sample Data

You can use the provided `sample_asteroid_data.json` file to test the system:

1. Copy it to `real_asteroid_data.json`
2. Run the training script
3. Start the API and test predictions

## Troubleshooting

- **Model files not found**: Make sure to run `train_model.py` successfully first
- **JSON format errors**: Validate your JSON file syntax
- **Missing columns**: Ensure all required fields are present in your JSON data
- **Import errors**: Make sure all required packages are installed

## API Documentation

Once the API is running, visit http://localhost:8000/docs for interactive API documentation (Swagger UI).