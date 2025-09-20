"""
Integration tests for API endpoints.
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from predictionlayer.schemas import PredictionResponse, NEOPrediction, RiskBucket, RiskTerms


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_endpoint_healthy(self, test_client):
        """Test health endpoint returns healthy status."""
        response = test_client.get("/health/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "dependencies" in data
        
        # Should show degraded due to unhealthy query_layer mock
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert data["version"] == "1.0.0"
        assert "cache" in data["dependencies"]
        assert "query_layer" in data["dependencies"]
        assert "model_layer" in data["dependencies"]
    
    def test_health_endpoint_redirect(self, test_client):
        """Test health endpoint without trailing slash redirects."""
        response = test_client.get("/health")
        
        # FastAPI should redirect to /health/
        assert response.status_code == 307
        assert response.headers["location"] == "/health/"


class TestMetricsEndpoint:
    """Test metrics endpoint."""
    
    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint returns proper metrics."""
        response = test_client.get("/metrics/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all expected metrics are present
        expected_metrics = [
            "requests_total",
            "requests_per_minute", 
            "cache_hit_rate",
            "average_response_time_ms",
            "upstream_calls_total",
            "error_rate",
            "active_connections"
        ]
        
        for metric in expected_metrics:
            assert metric in data
            assert isinstance(data[metric], (int, float))
            assert data[metric] >= 0
    
    def test_metrics_endpoint_redirect(self, test_client):
        """Test metrics endpoint without trailing slash redirects."""
        response = test_client.get("/metrics")
        
        assert response.status_code == 307
        assert response.headers["location"] == "/metrics/"


class TestPredictionsEndpoint:
    """Test predictions API endpoint."""
    
    @pytest.fixture
    def mock_prediction_data(self):
        """Mock prediction data for testing."""
        return {
            "predictions": [
                {
                    "designation": "2024 AB00",
                    "des_key": "2024_AB00",
                    "diameter_km": 4.01,
                    "absolute_magnitude": 27.15,
                    "albedo": None,
                    "close_approach_date": "2024-01-01T00:00:00Z",
                    "distance_au": 0.036,
                    "velocity_km_s": 41.36,
                    "impact_probability": 0.0008,
                    "palermo_scale": -3.46,
                    "torino_scale": 2,
                    "potential_impacts": 58,
                    "risk_score": 0.983,
                    "risk_bucket": "high",
                    "risk_terms": {
                        "diameter_cubed": 64.47,
                        "velocity_squared": 1710.49,
                        "inverse_distance": 27.08
                    },
                    "score_notes": "Mock calculation",
                    "is_pha": False,
                    "data_sources": ["CAD", "Mock"]
                }
            ],
            "pagination": {
                "page": 1,
                "page_size": 10,
                "total_rows": 1,
                "total_pages": 1,
                "has_next": False,
                "has_prev": False
            },
            "generated_at": "2024-01-01T12:00:00Z",
            "cache_hit": False,
            "etag": "test_etag",
            "summary": {"total_objects": 1}
        }
    
    def test_predictions_endpoint_valid_request(self, test_client, mock_cache_client, mock_prediction_data):
        """Test predictions endpoint with valid request."""
        # Mock the cache to return None (cache miss)
        mock_cache_client.get.return_value = None
        
        request_data = {
            "date_min": "2024-01-01T00:00:00Z",
            "date_max": "2024-01-31T23:59:59Z",
            "dist_max_au": 0.05,
            "page": 1,
            "page_size": 10
        }
        
        # Mock the generate_predictions_data function to return our mock data
        with patch('predictionlayer.routers.predictions.generate_predictions_data') as mock_generate:
            mock_generate.return_value = mock_prediction_data
            
            response = test_client.post("/api/v1/predictions/", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "predictions" in data
        assert "pagination" in data
        assert "generated_at" in data
        assert "cache_hit" in data
        assert "etag" in data
        assert "summary" in data
        
        # Verify prediction data
        assert len(data["predictions"]) == 1
        prediction = data["predictions"][0]
        assert prediction["designation"] == "2024 AB00"
        assert prediction["risk_bucket"] == "high"
        assert prediction["risk_score"] == 0.983
        
        # Verify pagination
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["total_rows"] == 1
        assert data["pagination"]["has_next"] is False
    
    def test_predictions_endpoint_validation_errors(self, test_client):
        """Test predictions endpoint with invalid request data."""
        # Missing required fields
        response = test_client.post("/api/v1/predictions/", json={})
        
        assert response.status_code == 422
        errors = response.json()["detail"]
        
        # Should have errors for missing date_min and date_max
        error_fields = [error["loc"][-1] for error in errors]
        assert "date_min" in error_fields
        assert "date_max" in error_fields
    
    def test_predictions_endpoint_invalid_date_range(self, test_client):
        """Test predictions endpoint with invalid date range."""
        request_data = {
            "date_min": "2024-01-31T00:00:00Z",
            "date_max": "2024-01-01T00:00:00Z",  # End before start
        }
        
        response = test_client.post("/api/v1/predictions/", json=request_data)
        
        assert response.status_code == 422
        errors = response.json()["detail"]
        
        # Should have validation error about date range
        assert any("date_max must be after date_min" in str(error) for error in errors)
    
    def test_predictions_endpoint_invalid_distance(self, test_client):
        """Test predictions endpoint with invalid distance values."""
        request_data = {
            "date_min": "2024-01-01T00:00:00Z",
            "date_max": "2024-01-31T23:59:59Z",
            "dist_max_au": 2.0,  # Too large
        }
        
        response = test_client.post("/api/v1/predictions/", json=request_data)
        
        assert response.status_code == 422
    
    def test_predictions_endpoint_pagination(self, test_client):
        """Test predictions endpoint pagination parameters."""
        request_data = {
            "date_min": "2024-01-01T00:00:00Z",
            "date_max": "2024-01-31T23:59:59Z",
            "page": 2,
            "page_size": 5
        }
        
        with patch('predictionlayer.routers.predictions.generate_predictions_data') as mock_generate:
            mock_generate.return_value = {
                "predictions": [],
                "pagination": {
                    "page": 2,
                    "page_size": 5,
                    "total_rows": 10,
                    "total_pages": 2,
                    "has_next": False,
                    "has_prev": True
                },
                "generated_at": "2024-01-01T12:00:00Z",
                "cache_hit": False,
                "etag": "test_etag",
                "summary": {}
            }
            
            response = test_client.post("/api/v1/predictions/", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["pagination"]["page"] == 2
        assert data["pagination"]["page_size"] == 5
        assert data["pagination"]["has_prev"] is True
    
    def test_predictions_endpoint_cache_hit(self, test_client, mock_cache_client):
        """Test predictions endpoint with cache hit."""
        cached_data = {
            "predictions": [],
            "total_rows": 0,
            "etag": "cached_etag"
        }
        
        mock_cache_client.get.return_value = cached_data
        
        request_data = {
            "date_min": "2024-01-01T00:00:00Z",
            "date_max": "2024-01-31T23:59:59Z"
        }
        
        response = test_client.post("/api/v1/predictions/", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should indicate cache hit
        assert data["cache_hit"] is True
        assert data["etag"] == "cached_etag"
    
    def test_predictions_endpoint_etag_not_modified(self, test_client, mock_cache_client):
        """Test predictions endpoint with ETag returning 304 Not Modified."""
        cached_data = {
            "predictions": [],
            "total_rows": 0,
            "etag": "test_etag"
        }
        
        mock_cache_client.get.return_value = cached_data
        
        request_data = {
            "date_min": "2024-01-01T00:00:00Z",
            "date_max": "2024-01-31T23:59:59Z"
        }
        
        # Send request with If-None-Match header
        response = test_client.post(
            "/api/v1/predictions/", 
            json=request_data,
            headers={"If-None-Match": "test_etag"}
        )
        
        assert response.status_code == 304


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_docs_accessible(self, test_client):
        """Test that OpenAPI documentation is accessible."""
        response = test_client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_openapi_json_accessible(self, test_client):
        """Test that OpenAPI JSON schema is accessible."""
        response = test_client.get("/openapi.json")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Verify it's valid JSON
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Verify our endpoints are documented
        paths = schema["paths"]
        assert "/health/" in paths
        assert "/metrics/" in paths
        assert "/api/v1/predictions/" in paths


class TestErrorHandling:
    """Test error handling across endpoints."""
    
    def test_404_not_found(self, test_client):
        """Test 404 handling for non-existent endpoints."""
        response = test_client.get("/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_method_not_allowed(self, test_client):
        """Test 405 handling for wrong HTTP methods."""
        # Try GET on predictions endpoint (should be POST)
        response = test_client.get("/api/v1/predictions/")
        
        assert response.status_code == 405
    
    def test_predictions_endpoint_server_error(self, test_client):
        """Test predictions endpoint handling server errors."""
        request_data = {
            "date_min": "2024-01-01T00:00:00Z",
            "date_max": "2024-01-31T23:59:59Z"
        }
        
        # Mock the function to raise an exception
        with patch('predictionlayer.routers.predictions.generate_predictions_data') as mock_generate:
            mock_generate.side_effect = Exception("Test error")
            
            response = test_client.post("/api/v1/predictions/", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        
        assert "error" in data
        assert "message" in data
        assert "request_id" in data