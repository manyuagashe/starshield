"""
Unit tests for Pydantic schemas and data validation.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from predictionlayer.schemas import (
    PredictionRequest,
    PredictionResponse,
    NEOPrediction,
    RiskBucket,
    RiskTerms,
    PaginationInfo,
    HealthResponse,
    MetricsResponse,
)


class TestPredictionRequest:
    """Test PredictionRequest schema validation."""
    
    def test_valid_prediction_request(self):
        """Test valid prediction request creation."""
        request = PredictionRequest(
            date_min=datetime(2024, 1, 1, tzinfo=timezone.utc),
            date_max=datetime(2024, 1, 31, tzinfo=timezone.utc),
            dist_max_au=0.05,
            page=1,
            page_size=50,
        )
        
        assert request.date_min.year == 2024
        assert request.date_min.month == 1
        assert request.date_max.day == 31
        assert request.dist_max_au == 0.05
        assert request.page == 1
        assert request.page_size == 50
        
    def test_prediction_request_defaults(self):
        """Test prediction request with default values."""
        request = PredictionRequest(
            date_min=datetime(2024, 1, 1, tzinfo=timezone.utc),
            date_max=datetime(2024, 1, 31, tzinfo=timezone.utc),
        )
        
        assert request.dist_max_au == 0.05
        assert request.ip_min == 1e-8
        assert request.include_neows is True
        assert request.pha_only is False
        assert request.page == 1
        assert request.page_size == 50
        
    def test_invalid_date_range(self):
        """Test validation fails for invalid date range."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(
                date_min=datetime(2024, 1, 31, tzinfo=timezone.utc),
                date_max=datetime(2024, 1, 1, tzinfo=timezone.utc),  # End before start
            )
        
        errors = exc_info.value.errors()
        assert any("date_max must be after date_min" in str(error) for error in errors)
        
    def test_invalid_distance_range(self):
        """Test validation fails for invalid distance range."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(
                date_min=datetime(2024, 1, 1, tzinfo=timezone.utc),
                date_max=datetime(2024, 1, 31, tzinfo=timezone.utc),
                dist_max_au=0.0001,  # Too small
            )
        
        # Should pass validation as 0.0001 is >= 0.001
        # Let's test a value that's actually invalid
        with pytest.raises(ValidationError):
            PredictionRequest(
                date_min=datetime(2024, 1, 1, tzinfo=timezone.utc),
                date_max=datetime(2024, 1, 31, tzinfo=timezone.utc),
                dist_max_au=2.0,  # Too large (> 1.0)
            )
        
    def test_negative_values_validation(self):
        """Test validation fails for negative values."""
        with pytest.raises(ValidationError):
            PredictionRequest(
                date_min=datetime(2024, 1, 1, tzinfo=timezone.utc),
                date_max=datetime(2024, 1, 31, tzinfo=timezone.utc),
                dist_max_au=-0.1,  # Negative distance
            )
            
    def test_page_validation(self):
        """Test page and page_size validation."""
        with pytest.raises(ValidationError):
            PredictionRequest(
                date_min=datetime(2024, 1, 1, tzinfo=timezone.utc),
                date_max=datetime(2024, 1, 31, tzinfo=timezone.utc),
                page=0,  # Page must be >= 1
            )
            
        with pytest.raises(ValidationError):
            PredictionRequest(
                date_min=datetime(2024, 1, 1, tzinfo=timezone.utc),
                date_max=datetime(2024, 1, 31, tzinfo=timezone.utc),
                page_size=1001,  # Page size too large
            )


class TestNEOPrediction:
    """Test NEOPrediction schema."""
    
    def test_valid_neo_prediction(self):
        """Test valid NEO prediction creation."""
        prediction = NEOPrediction(
            designation="2024 AB00",
            des_key="2024_AB00",
            diameter_km=4.0,
            absolute_magnitude=27.1,
            albedo=0.15,
            close_approach_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            distance_au=0.036,
            velocity_km_s=41.36,
            impact_probability=0.0001,
            palermo_scale=-3.46,
            torino_scale=2,
            potential_impacts=58,
            risk_score=0.983,
            risk_bucket=RiskBucket.HIGH,
            risk_terms=RiskTerms(
                diameter_cubed=64.0,
                velocity_squared=1710.0,
                inverse_distance=27.0,
            ),
            score_notes="Test calculation",
            is_pha=False,
            data_sources=["CAD", "Mock"],
        )
        
        assert prediction.designation == "2024 AB00"
        assert prediction.des_key == "2024_AB00"
        assert prediction.diameter_km == 4.0
        assert prediction.risk_bucket == RiskBucket.HIGH
        assert prediction.is_pha is False
        assert "CAD" in prediction.data_sources
        
    def test_neo_prediction_optional_fields(self):
        """Test NEO prediction with optional fields as None."""
        prediction = NEOPrediction(
            designation="2024 AB00",
            des_key="2024_AB00",
            diameter_km=None,
            absolute_magnitude=None,
            albedo=None,
            close_approach_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            distance_au=0.036,
            velocity_km_s=41.36,
            impact_probability=None,
            palermo_scale=None,
            torino_scale=None,
            potential_impacts=None,
            risk_score=0.5,
            risk_bucket=RiskBucket.MEDIUM,
            risk_terms=RiskTerms(
                diameter_cubed=0.0,
                velocity_squared=1710.0,
                inverse_distance=27.0,
            ),
            score_notes="",
            is_pha=False,
            data_sources=["CAD"],
        )
        
        assert prediction.diameter_km is None
        assert prediction.absolute_magnitude is None
        assert prediction.albedo is None
        assert prediction.impact_probability is None


class TestRiskBucket:
    """Test RiskBucket enumeration."""
    
    def test_risk_bucket_values(self):
        """Test all risk bucket values."""
        assert RiskBucket.LOW == "low"
        assert RiskBucket.MEDIUM == "medium"
        assert RiskBucket.HIGH == "high"
        
    def test_risk_bucket_in_neo_prediction(self):
        """Test risk bucket in NEO prediction."""
        for bucket in [RiskBucket.LOW, RiskBucket.MEDIUM, RiskBucket.HIGH]:
            prediction = NEOPrediction(
                designation="Test",
                des_key="test",
                diameter_km=None,
                absolute_magnitude=None,
                albedo=None,
                close_approach_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                distance_au=0.1,
                velocity_km_s=20.0,
                impact_probability=None,
                palermo_scale=None,
                torino_scale=None,
                potential_impacts=None,
                risk_score=0.5,
                risk_bucket=bucket,
                risk_terms=RiskTerms(
                    diameter_cubed=1.0,
                    velocity_squared=400.0,
                    inverse_distance=10.0,
                ),
                score_notes="",
                is_pha=False,
                data_sources=["Test"],
            )
            assert prediction.risk_bucket == bucket


class TestRiskTerms:
    """Test RiskTerms schema."""
    
    def test_risk_terms_creation(self):
        """Test risk terms creation and validation."""
        terms = RiskTerms(
            diameter_cubed=64.0,
            velocity_squared=1710.0,
            inverse_distance=27.0,
        )
        
        assert terms.diameter_cubed == 64.0
        assert terms.velocity_squared == 1710.0
        assert terms.inverse_distance == 27.0
        
    def test_risk_terms_negative_values(self):
        """Test risk terms with edge cases."""
        # Zero values should be allowed
        terms = RiskTerms(
            diameter_cubed=0.0,
            velocity_squared=0.0,
            inverse_distance=0.0,
        )
        
        assert terms.diameter_cubed == 0.0
        assert terms.velocity_squared == 0.0
        assert terms.inverse_distance == 0.0


class TestPaginationInfo:
    """Test PaginationInfo schema."""
    
    def test_pagination_info(self):
        """Test pagination info creation."""
        pagination = PaginationInfo(
            page=2,
            page_size=50,
            total_rows=150,
            total_pages=3,
            has_next=True,
            has_prev=True,
        )
        
        assert pagination.page == 2
        assert pagination.page_size == 50
        assert pagination.total_rows == 150
        assert pagination.total_pages == 3
        assert pagination.has_next is True
        assert pagination.has_prev is True
        
    def test_pagination_first_page(self):
        """Test pagination for first page."""
        pagination = PaginationInfo(
            page=1,
            page_size=50,
            total_rows=150,
            total_pages=3,
            has_next=True,
            has_prev=False,
        )
        
        assert pagination.has_prev is False
        assert pagination.has_next is True
        
    def test_pagination_last_page(self):
        """Test pagination for last page."""
        pagination = PaginationInfo(
            page=3,
            page_size=50,
            total_rows=150,
            total_pages=3,
            has_next=False,
            has_prev=True,
        )
        
        assert pagination.has_next is False
        assert pagination.has_prev is True


class TestPredictionResponse:
    """Test PredictionResponse schema."""
    
    def test_prediction_response_creation(self, sample_neo_data):
        """Test prediction response with mock data."""
        predictions = [
            NEOPrediction(
                designation=neo["designation"],
                des_key=neo["des_key"],
                diameter_km=neo.get("diameter_km"),
                absolute_magnitude=neo.get("absolute_magnitude"),
                albedo=None,
                close_approach_date=neo["close_approach_date"],
                distance_au=neo["distance_au"],
                velocity_km_s=neo["velocity_km_s"],
                impact_probability=None,
                palermo_scale=None,
                torino_scale=None,
                potential_impacts=None,
                risk_score=0.5,
                risk_bucket=RiskBucket.MEDIUM,
                risk_terms=RiskTerms(
                    diameter_cubed=1.0,
                    velocity_squared=400.0,
                    inverse_distance=10.0,
                ),
                score_notes="Test",
                is_pha=neo["is_pha"],
                data_sources=["Test"],
            )
            for neo in sample_neo_data
        ]
        
        pagination = PaginationInfo(
            page=1,
            page_size=10,
            total_rows=2,
            total_pages=1,
            has_next=False,
            has_prev=False,
        )
        
        response = PredictionResponse(
            predictions=predictions,
            pagination=pagination,
            generated_at=datetime.now(timezone.utc),
            cache_hit=False,
            etag="test_etag",
            summary={"test": "data"},
        )
        
        assert len(response.predictions) == 2
        assert response.pagination.total_rows == 2
        assert response.cache_hit is False
        assert response.etag == "test_etag"
        assert response.summary == {"test": "data"}


class TestHealthResponse:
    """Test HealthResponse schema."""
    
    def test_health_response_creation(self):
        """Test health response creation."""
        health = HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            dependencies={
                "cache": "healthy",
                "query_layer": "degraded",
                "model_layer": "healthy"
            }
        )
        
        assert health.status == "healthy"
        assert health.version == "1.0.0"
        assert health.dependencies["cache"] == "healthy"
        assert health.dependencies["query_layer"] == "degraded"


class TestMetricsResponse:
    """Test MetricsResponse schema."""
    
    def test_metrics_response_creation(self):
        """Test metrics response creation."""
        metrics = MetricsResponse(
            requests_total=1000,
            requests_per_minute=50.0,
            cache_hit_rate=0.85,
            average_response_time_ms=150.5,
            upstream_calls_total=200,
            error_rate=0.02,
            active_connections=10
        )
        
        assert metrics.requests_total == 1000
        assert metrics.requests_per_minute == 50.0
        assert metrics.cache_hit_rate == 0.85
        assert metrics.average_response_time_ms == 150.5
        assert metrics.upstream_calls_total == 200
        assert metrics.error_rate == 0.02
        assert metrics.active_connections == 10