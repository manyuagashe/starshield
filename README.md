# StarShield - NASA CAD Asteroid Risk Prediction System

A production-ready asteroid risk prediction system built on real NASA JPL Close Approach Data (CAD).

## 🌟 Overview

StarShield uses real NASA data to predict asteroid risk levels using machine learning. The entire system is built around NASA's JPL Small-Body Database Close Approach Data API, providing accurate risk assessments for Near-Earth Objects (NEOs).

## 🚀 Features

- **Real NASA Data**: Built on NASA JPL SBDB Close Approach Data API v1.5
- **Machine Learning**: RandomForest classifier trained on 48 real asteroid encounters
- **Production API**: Enhanced FastAPI with WebSocket, batch processing, and monitoring
- **Real-time Predictions**: Sub-second response times with confidence scoring
- **Comprehensive Monitoring**: Health checks, metrics, and connection tracking

## 📊 Architecture

```
NASA JPL CAD API → raw_data/cad.json → convert_cad_data.py → real_asteroid_data.json
                                                                      ↓
                                                              train_model.py
                                                                      ↓
                                                        asteroid_risk_model.joblib
                                                                      ↓
                                                              asteroid_api.py (FastAPI)
```

## 🔧 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Process NASA Data (Already Done)
```bash
# Convert NASA CAD data to training format
python convert_cad_data.py
```

### 3. Train Model (Already Done)
```bash
# Train RandomForest model on NASA data
python train_model.py
```

### 4. Start API
```bash
# Start the FastAPI server
python asteroid_api.py
```

The API will be available at: http://localhost:8000

## 📁 Core Files

### Data Pipeline
- **`raw_data/cad.json`** - NASA JPL Close Approach Data (48 asteroids)
- **`convert_cad_data.py`** - Converts NASA data to training format
- **`real_asteroid_data.json`** - Processed training data
- **`train_model.py`** - Model training script

### Trained Model
- **`asteroid_risk_model.joblib`** - RandomForest classifier
- **`risk_level_labels.joblib`** - Risk level mappings
- **`model_metadata.json`** - Training metadata

### API Service
- **`asteroid_api.py`** - Production FastAPI application

## 🔍 API Endpoints

### Health & Info
- `GET /` - System health and status
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe
- `GET /model/info` - Model information and metadata
- `GET /metrics` - System metrics

### Predictions
- `POST /predict` - Single asteroid risk prediction
- `POST /predict/batch` - Batch predictions (up to 100)
- `WebSocket /ws` - Real-time prediction streaming

### Sample Request
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

### Sample Response
```json
{
  "predicted_risk_level": "Medium",
  "predicted_risk_score": 0.4521,
  "confidence": 0.61,
  "prediction_probabilities": {
    "Low": 0.01,
    "Medium": 0.61,
    "High": 0.38,
    "Critical": 0.0
  },
  "prediction_id": "uuid-string",
  "timestamp": "2025-09-20T15:30:00",
  "processing_time_ms": 23.45
}
```

## 📈 Model Details

- **Training Data**: 48 real NASA asteroid close approaches
- **Algorithm**: RandomForest Classifier (100 estimators, max_depth=10)
- **Features**: Distance, velocity, diameter, PHA status, orbit class
- **Risk Levels**: Low, Medium, High, Critical
- **Accuracy**: 90% on test set

### Feature Importance
1. **distance_au** (41.5%) - Closest approach distance
2. **velocity_kms** (20.5%) - Relative velocity
3. **v_infinity_kms** (19.7%) - Velocity at infinity
4. **diameter_km** (12.7%) - Estimated diameter
5. **is_pha** (5.4%) - Potentially Hazardous Asteroid status

## 🔧 Development

### Interactive API Documentation
Visit http://localhost:8000/docs when the server is running

### Adding New Data
1. Update `raw_data/cad.json` with new NASA CAD data
2. Run `python convert_cad_data.py` to process new data
3. Run `python train_model.py` to retrain the model
4. Restart the API server

### WebSocket Usage
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.send(JSON.stringify({
  type: 'predict',
  asteroid: {
    distance_au: 0.018,
    velocity_kms: 22.3,
    // ... other fields
  }
}));
```

## 📊 Data Sources

- **NASA JPL SBDB Close Approach Data API v1.5**
- Real asteroid close approach events
- Orbital and physical characteristics
- Absolute magnitudes for diameter estimation

## 🚀 Production Features

- **CORS middleware** for web integration
- **Rate limiting** and connection tracking
- **Background task processing** for logging
- **Comprehensive error handling**
- **Structured logging** with timestamps
- **Health monitoring** and metrics
- **WebSocket support** for real-time updates

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

Built with ❤️ using real NASA data for asteroid risk assessment
