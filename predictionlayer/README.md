# StarShield Prediction Layer

The FastAPI wrapper component of the StarShield NEO risk assessment system.

## Overview

This layer provides a RESTful API that orchestrates calls between the Query layer (NASA API data retrieval) and Model layer (risk computation), with caching, rate limiting, and comprehensive error handling.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │────│  Prediction      │────│  Query Layer    │
│   (React/Vue)   │    │  Layer (FastAPI) │    │  (NASA APIs)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                │
                        ┌──────────────────┐
                        │   Model Layer    │
                        │  (Risk Compute)  │
                        └──────────────────┘
```

## Features

- **FastAPI** with async endpoints and automatic OpenAPI documentation
- **Pydantic** schemas for request/response validation
- **Caching** with Redis or SQLite backends, ETag support
- **Rate limiting** with token bucket algorithm
- **Metrics** collection for monitoring (Prometheus compatible)
- **Error handling** with structured error responses
- **Circuit breaker** pattern for upstream service protection
- **Health checks** for monitoring and orchestration

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Edit .env with your settings
```

## Configuration

Key environment variables (all prefixed with `STARSHIELD_`):

- `HOST`, `PORT` - Server binding
- `CACHE_BACKEND` - `redis` or `sqlite`
- `REDIS_URL` - Redis connection string
- `NASA_API_KEY` - NASA API key (or `DEMO_KEY`)
- `RATE_LIMIT_REQUESTS` - Requests per minute per IP
- `MAX_DATE_RANGE_DAYS` - Maximum query date range

## Running

### Development
```bash
# Using uvicorn directly
uvicorn predictionlayer.main:app --reload --port 8000

# Using the module
python -m predictionlayer.main
```

### Production
```bash
# Using gunicorn
gunicorn predictionlayer.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## API Endpoints

### Core Endpoints

- `POST /api/v1/predictions/` - Generate NEO risk predictions
- `GET /api/v1/predictions/summary` - Quick summary statistics

### Monitoring

- `GET /health/` - Comprehensive health check
- `GET /health/liveness` - Simple liveness probe  
- `GET /health/readiness` - Readiness probe
- `GET /metrics/` - Application metrics (JSON)
- `GET /metrics/prometheus` - Prometheus format metrics

### Documentation

- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative documentation (ReDoc)

## Request/Response Examples

### Get Predictions

```http
POST /api/v1/predictions/
Content-Type: application/json

{
  "date_min": "2024-01-01T00:00:00Z",
  "date_max": "2024-01-31T23:59:59Z", 
  "dist_max_au": 0.05,
  "ip_min": 1e-8,
  "include_neows": true,
  "pha_only": false,
  "page": 1,
  "page_size": 50
}
```

```http
200 OK
Content-Type: application/json
ETag: "a1b2c3d4e5f6"

{
  "predictions": [
    {
      "designation": "2024 AA1",
      "des_key": "2024_AA1",
      "diameter_km": 0.15,
      "close_approach_date": "2024-01-15T12:30:00Z",
      "distance_au": 0.02,
      "velocity_km_s": 18.5,
      "risk_score": 0.65,
      "risk_bucket": "medium",
      "risk_terms": {
        "diameter_cubed": 0.003375,
        "velocity_squared": 342.25,
        "inverse_distance": 50.0
      },
      "is_pha": false,
      "data_sources": ["CAD", "Sentry"]
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 50,
    "total_rows": 127,
    "total_pages": 3,
    "has_next": true,
    "has_prev": false
  },
  "generated_at": "2024-01-01T10:30:00Z",
  "cache_hit": false,
  "etag": "a1b2c3d4e5f6",
  "summary": {
    "risk_distribution": {"low": 95, "medium": 28, "high": 4},
    "date_range_days": 31,
    "total_close_approaches": 127
  }
}
```

## Integration with Other Layers

### Query Layer Integration

The prediction layer expects the Query layer to implement:

```python
class QueryClient(ABC):
    async def get_cad_data(self, date_min, date_max, dist_max_au, **kwargs) -> pd.DataFrame
    async def get_sentry_data(self, date_min, date_max, ip_min, **kwargs) -> pd.DataFrame  
    async def get_neows_data(self, date_min, date_max, **kwargs) -> Optional[pd.DataFrame]
```

### Model Layer Integration

The prediction layer expects the Model layer to implement:

```python
class ModelService(ABC):
    async def compute_risk_predictions(self, cad_data, sentry_data, neows_data, model_params) -> pd.DataFrame
    def get_model_params(self) -> Dict
    def validate_data_quality(self, df) -> Dict
```

## Caching Strategy

- **Key Generation**: Deterministic hashing of request parameters
- **TTL**: Configurable (default 6 hours)
- **ETag Support**: Conditional requests with `If-None-Match` headers
- **Backends**: Redis (production) or SQLite (development)

## Monitoring & Observability

### Metrics Collected

- Request count, rate, and latency
- Cache hit/miss rates
- Upstream API call counts
- Error rates by endpoint and status code
- Circuit breaker state

### Health Checks

- `/health/` - Full dependency health check
- `/health/liveness` - Basic service health
- `/health/readiness` - Ready to serve traffic

## Error Handling

Structured error responses with appropriate HTTP status codes:

```json
{
  "error": "validation_error",
  "message": "date_max must be after date_min", 
  "details": {
    "field": "date_max",
    "provided": "2024-01-01",
    "constraint": "must be after date_min"
  },
  "request_id": "req_123456789"
}
```

## Development

### Running Tests

```bash
pytest tests/ -v --cov=predictionlayer
```

### Code Quality

```bash
# Format code
black predictionlayer/
isort predictionlayer/

# Type checking
mypy predictionlayer/

# Linting
flake8 predictionlayer/
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY predictionlayer/ ./predictionlayer/
EXPOSE 8000

CMD ["uvicorn", "predictionlayer.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

Set all `STARSHIELD_*` environment variables for production deployment.

## Contributing

1. Follow the established patterns for new endpoints
2. Add appropriate error handling and logging
3. Include tests for new functionality
4. Update documentation for API changes
5. Ensure type hints and docstrings are present