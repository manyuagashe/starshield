# 🌟 StarShield Asteroid Risk Prediction API

A production-ready FastAPI service that predicts Near-Earth Object (NEO) risk levels using a RandomForest model trained on real NASA JPL Close Approach Data (CAD).

## 🚀 Overview

The StarShield API provides real-time asteroid risk assessment with high accuracy predictions based on orbital characteristics. The system uses machine learning to classify asteroids into risk categories: **Low**, **Medium**, **High**, and **Critical**.

### Key Features
- ✅ **Real NASA Data**: Trained on 1,300 samples from 13,321+ NASA JPL CAD records
- ✅ **High Accuracy**: 99.62% accuracy on test data
- ✅ **Fast Predictions**: ~30-60ms response times
- ✅ **Batch Processing**: Handle multiple asteroids in single request  
- ✅ **WebSocket Support**: Real-time streaming predictions
- ✅ **Health Monitoring**: Comprehensive API health endpoints
- ✅ **Production Ready**: CORS, error handling, logging, and monitoring

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Training Data** | 1,040 samples (80%) |
| **Test Data** | 260 samples (20%) |
| **Overall Accuracy** | 99.62% |
| **Medium Risk Accuracy** | 100.0% |
| **Low Risk Accuracy** | 100.0% |
| **High Risk Accuracy** | 92.3% |

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Quick Start
```bash
# Clone and setup
git clone <repository>
cd starshield

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate training data (1300 samples from NASA data)
python convert_cad_data.py

# Train the model
python train_model.py

# Start the API server
uvicorn asteroid_api:app --host 0.0.0.0 --port 8000
```

The API will be available at: **http://localhost:8000**

Interactive docs: **http://localhost:8000/docs**

## 📡 API Endpoints

### Health & Monitoring

#### `GET /` - Root Health Check
```bash
curl http://localhost:8000/
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0",
  "uptime_seconds": 232.16,
  "total_predictions": 1303,
  "current_connections": 1
}
```

#### `GET /health/live` - Liveness Probe
```bash
curl http://localhost:8000/health/live
```
```json
{
  "status": "alive",
  "timestamp": "2025-09-20T17:04:28.512078"
}
```

#### `GET /model/info` - Model Information
```bash
curl http://localhost:8000/model/info
```
```json
{
  "model_type": "RandomForestClassifier",
  "training_records": 1040,
  "features_used": 9,
  "available_classes": ["Low", "Medium", "High", "Critical"],
  "model_version": "1.0.0",
  "training_data_source": "NASA JPL CAD API"
}
```

### Prediction Endpoints

#### `POST /predict` - Single Asteroid Prediction
Predict risk level for a single asteroid.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "distance_au": 0.015,
    "velocity_kms": 8.5,  
    "diameter_km": 0.025,
    "v_infinity_kms": 12.3,
    "is_pha": false,
    "orbit_class": "AMO"
  }'
```

**Response:**
```json
{
  "input_features": {
    "distance_au": 0.015,
    "velocity_kms": 8.5,
    "diameter_km": 0.025,
    "v_infinity_kms": 12.3,
    "is_pha": false,
    "orbit_class": "AMO"
  },
  "predicted_risk_level": "Medium",
  "predicted_risk_score": 0.3267,
  "confidence": 0.99,
  "prediction_probabilities": {
    "Low": 0.01,
    "Medium": 0.99,
    "High": 0.0,
    "Critical": 0.0
  },
  "model_info": {
    "model_type": "RandomForestClassifier",
    "features_used": 9,
    "available_classes": ["Low", "Medium", "High", "Critical"]
  },
  "prediction_id": "50fc9d79-e41f-4785-b672-fd282db4f400",
  "timestamp": "2025-09-20T17:11:38.631764",
  "processing_time_ms": 41.57
}
```

#### `POST /predict/batch` - Batch Predictions
Process multiple asteroids in a single request (max 100).

**Request:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "asteroids": [
      {
        "distance_au": 0.015,
        "velocity_kms": 8.5,
        "diameter_km": 0.025,  
        "v_infinity_kms": 12.3,
        "is_pha": false,
        "orbit_class": "AMO"
      },
      {
        "distance_au": 0.008,
        "velocity_kms": 15.2,
        "diameter_km": 0.045,
        "v_infinity_kms": 18.7, 
        "is_pha": true,
        "orbit_class": "ATE"
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "input_features": {...},
      "predicted_risk_level": "Medium",
      "predicted_risk_score": 0.3267,
      "confidence": 0.99,
      "prediction_probabilities": {...},
      "model_info": {...},
      "prediction_id": "batch_001_item_0",
      "timestamp": "2025-09-20T17:11:38.631764",
      "processing_time_ms": 41.57
    }
  ],
  "batch_id": "batch_20250920_171138",
  "total_predictions": 2,
  "successful_predictions": 2,
  "failed_predictions": 0,
  "total_processing_time_ms": 60.14
}
```

### Data Endpoints

#### `GET /data/train` - Training Data Predictions
Get predictions for all 1,040 training samples.

```bash
curl http://localhost:8000/data/train
```

Returns batch prediction response with all training data predictions.

#### `GET /data/test` - Test Data Predictions  
Get predictions for all 260 test samples.

```bash
curl http://localhost:8000/data/test
```

Returns batch prediction response with all test data predictions.

## 📋 Input Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `distance_au` | float | Minimum approach distance in AU | `0.015` |
| `velocity_kms` | float | Relative velocity in km/s | `8.5` |
| `diameter_km` | float | Estimated diameter in kilometers | `0.025` |
| `v_infinity_kms` | float | Velocity at infinity in km/s | `12.3` |
| `is_pha` | boolean | Potentially Hazardous Asteroid flag | `false` |
| `orbit_class` | string | Orbit classification | `"AMO"`, `"ATE"`, `"APO"`, `"IEO"` |

### Orbit Classes
- **AMO**: Amor - Earth-approaching, semi-major axis > 1.0 AU
- **ATE**: Aten - Earth-crossing, semi-major axis < 1.0 AU  
- **APO**: Apollo - Earth-crossing, semi-major axis > 1.0 AU
- **IEO**: Interior Earth Object - Orbits entirely within Earth's orbit

## 🎯 Risk Levels

| Risk Level | Description | Criteria |
|------------|-------------|----------|
| **Low** | Minimal threat | Large distance, moderate velocity, small size |
| **Medium** | Moderate concern | Close approach with moderate characteristics |
| **High** | Significant risk | Close approach with high velocity or large size |
| **Critical** | Immediate threat | Very close approach with dangerous characteristics |

## 🔍 Response Fields

### Prediction Response
- `input_features`: Original input parameters
- `predicted_risk_level`: Risk classification (Low/Medium/High/Critical)
- `predicted_risk_score`: Numerical risk score (0-1)
- `confidence`: Model confidence in prediction (0-1)
- `prediction_probabilities`: Probability for each risk class
- `model_info`: Model metadata and configuration
- `prediction_id`: Unique identifier for this prediction
- `timestamp`: ISO 8601 timestamp
- `processing_time_ms`: Processing time in milliseconds

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_api.py
```

Test specific endpoints:
```bash
# Health check
curl http://localhost:8000/health/live

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_au": 0.015, "velocity_kms": 8.5, "diameter_km": 0.025, "v_infinity_kms": 12.3, "is_pha": false, "orbit_class": "AMO"}'

# Batch prediction  
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"asteroids": [{"distance_au": 0.015, "velocity_kms": 8.5, "diameter_km": 0.025, "v_infinity_kms": 12.3, "is_pha": false, "orbit_class": "AMO"}]}'
```

## 📊 Monitoring & Metrics

The API provides comprehensive monitoring:

#### `GET /metrics` - System Metrics
```json
{
  "total_predictions": 1303,
  "uptime_seconds": 232.16,
  "current_connections": 1,
  "prediction_rate_per_minute": 5.6,
  "average_processing_time_ms": 45.2,
  "model_accuracy": 99.62,
  "memory_usage_mb": 156.7
}
```

## 🌐 WebSocket Support

Real-time predictions via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    distance_au: 0.015,
    velocity_kms: 8.5,
    diameter_km: 0.025,
    v_infinity_kms: 12.3,
    is_pha: false,
    orbit_class: "AMO"
  }));
};

ws.onmessage = (event) => {
  const prediction = JSON.parse(event.data);
  console.log('Risk Level:', prediction.predicted_risk_level);
};
```

## 🛡️ Security & Production

### Security Features
- CORS middleware configured
- Input validation with Pydantic models
- Request rate limiting capabilities
- Trusted host middleware
- Error handling and logging

### Production Deployment
```bash
# Using Gunicorn with Uvicorn workers
gunicorn asteroid_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker
docker build -t starshield-api .
docker run -p 8000:8000 starshield-api

# Environment variables
export LOG_LEVEL=INFO
export MAX_BATCH_SIZE=100
export MODEL_PATH=./asteroid_risk_model.joblib
```

## 📁 File Structure

```
starshield/
├── asteroid_api.py              # Main FastAPI application
├── convert_cad_data.py          # NASA data processing
├── train_model.py               # Model training script
├── test_api.py                  # Comprehensive test suite
├── requirements.txt             # Python dependencies
├── real_asteroid_data_train.json # Training data (1040 samples)
├── real_asteroid_data_test.json  # Test data (260 samples)
├── asteroid_risk_model.joblib    # Trained model
├── risk_level_labels.joblib      # Label encoder
├── model_metadata.json          # Model configuration
└── raw_data/
    └── cad.json                 # Original NASA CAD data (13,321 records)
```

## 🔄 Data Pipeline

1. **Data Collection**: NASA JPL CAD API data (13,321+ asteroid records)
2. **Sampling**: Random selection of 1,300 representative samples
3. **Train/Test Split**: 80% training (1,040) / 20% testing (260)
4. **Feature Engineering**: Orbital parameters, PHA classification, risk scoring
5. **Model Training**: RandomForest with 100 estimators, max_depth=10
6. **Model Evaluation**: 99.62% accuracy on test data
7. **API Deployment**: FastAPI service with real-time predictions

## 🔧 Troubleshooting

### Common Issues

**Model Not Loading**
```bash
# Ensure model files exist
ls -la *.joblib model_metadata.json

# Retrain if necessary  
python train_model.py
```

**Port Already in Use**
```bash
# Kill existing process
pkill -f "uvicorn asteroid_api:app"

# Use different port
uvicorn asteroid_api:app --port 8001
```

**Import Errors**  
```bash
# Install dependencies
pip install -r requirements.txt

# Activate virtual environment
source .venv/bin/activate
```

## 📚 References

- [NASA JPL Small-Body Database](https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html)
- [Close Approach Data API](https://ssd-api.jpl.nasa.gov/doc/cad.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

**🌟 StarShield - Protecting Earth through AI-powered asteroid risk assessment**