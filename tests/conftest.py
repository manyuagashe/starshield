"""
Shared pytest fixtures and configuration for StarShield tests.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from predictionlayer.main import create_app
from predictionlayer.config import Settings, get_settings
from predictionlayer.cache import CacheClient
from predictionlayer.dependencies import (
    QueryClient, 
    ModelService, 
    get_cache_client_dep,
    get_query_client_dep,
    get_model_service_dep
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Test configuration settings."""
    return Settings(
        debug=True,
        cache_backend="sqlite",
        redis_url="redis://localhost:6379/1",  # Test Redis DB
        cache_ttl_seconds=300,
        rate_limit_requests=1000,
        rate_limit_window_seconds=60,
    )


@pytest.fixture
def mock_cache_client():
    """Mock cache client for testing."""
    mock = AsyncMock(spec=CacheClient)
    mock.get.return_value = None
    mock.set.return_value = None
    mock.delete.return_value = None
    mock.generate_cache_key.return_value = "test_key"
    mock.generate_etag.return_value = "test_etag"
    return mock


@pytest.fixture
def mock_query_client():
    """Mock query client for testing."""
    mock = AsyncMock(spec=QueryClient)
    mock.is_healthy.return_value = False  # Simulate unhealthy for degraded status
    return mock


@pytest.fixture
def mock_model_service():
    """Mock model service for testing."""
    mock = AsyncMock(spec=ModelService)
    mock.is_healthy.return_value = True
    return mock


@pytest.fixture
def sample_neo_data():
    """Sample NEO data for testing."""
    return [
        {
            "designation": "2024 AB00",
            "des_key": "2024_AB00",
            "close_approach_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "distance_au": 0.036,
            "velocity_km_s": 41.36,
            "absolute_magnitude": 27.15,
            "diameter_km": 4.01,
            "is_pha": False,
        },
        {
            "designation": "2024 AB01", 
            "des_key": "2024_AB01",
            "close_approach_date": datetime(2024, 1, 2, tzinfo=timezone.utc),
            "distance_au": 0.034,
            "velocity_km_s": 16.64,
            "absolute_magnitude": 15.68,
            "diameter_km": 1.82,
            "is_pha": True,
        }
    ]


@pytest.fixture
def test_app(test_settings, mock_cache_client, mock_query_client, mock_model_service):
    """Test FastAPI application with mocked dependencies."""
    app = create_app()
    
    # Override dependencies
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_cache_client_dep] = lambda: mock_cache_client
    app.dependency_overrides[get_query_client_dep] = lambda: mock_query_client
    app.dependency_overrides[get_model_service_dep] = lambda: mock_model_service
    
    return app


@pytest.fixture
def test_client(test_app):
    """FastAPI test client."""
    return TestClient(test_app)


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request data."""
    return {
        "date_min": "2024-01-01T00:00:00Z",
        "date_max": "2024-01-31T23:59:59Z",
        "dist_min_au": 0.0,
        "dist_max_au": 1.0,
        "page": 1,
        "page_size": 10,
    }