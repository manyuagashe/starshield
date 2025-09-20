"""
Pydantic schemas for request/response validation and data models.

This module defines the data structures used for API validation,
serialization, and communication between layers.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field, field_validator


class RiskBucket(str, Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PredictionRequest(BaseModel):
    """Request schema for NEO risk predictions."""
    
    date_min: datetime = Field(
        description="Start date for the prediction window"
    )
    date_max: datetime = Field(
        description="End date for the prediction window"
    )
    dist_max_au: Optional[float] = Field(
        default=0.05,
        ge=0.001,
        le=1.0,
        description="Maximum distance in AU for filtering objects"
    )
    ip_min: Optional[float] = Field(
        default=1e-8,
        ge=1e-12,
        le=1.0,
        description="Minimum impact probability for filtering"
    )
    include_neows: bool = Field(
        default=True,
        description="Whether to include NeoWs data for diameter estimates"
    )
    pha_only: bool = Field(
        default=False,
        description="Filter to only Potentially Hazardous Asteroids"
    )
    page: int = Field(default=1, ge=1, description="Page number for pagination")
    page_size: int = Field(
        default=50, 
        ge=1, 
        le=1000, 
        description="Number of results per page"
    )
    
    @field_validator("date_max")
    @classmethod
    def validate_date_range(cls, v, info):
        """Ensure date_max is after date_min."""
        if hasattr(info, 'data') and info.data:
            date_min = info.data.get("date_min")
            if date_min and v <= date_min:
                raise ValueError("date_max must be after date_min")
            
            # Check maximum range (from config)
            if date_min and (v - date_min).days > 365:
                raise ValueError("Date range cannot exceed 365 days")
        
        return v


class RiskTerms(BaseModel):
    """Individual risk calculation components."""
    
    diameter_cubed: float = Field(description="D^3 component")
    velocity_squared: float = Field(description="v^2 component") 
    inverse_distance: float = Field(description="1/(dist+ε) component")


class NEOPrediction(BaseModel):
    """Individual NEO risk prediction result."""
    
    # Identifiers
    designation: str = Field(description="Object designation/name")
    des_key: str = Field(description="Normalized designation key for joins")
    
    # Orbit and physical properties
    diameter_km: Optional[float] = Field(description="Estimated diameter in km")
    absolute_magnitude: Optional[float] = Field(description="Absolute magnitude (H)")
    albedo: Optional[float] = Field(description="Geometric albedo")
    
    # Close approach data
    close_approach_date: datetime = Field(description="Date of close approach")
    distance_au: float = Field(description="Distance at close approach (AU)")
    velocity_km_s: float = Field(description="Relative velocity (km/s)")
    
    # Sentry risk data (if available)
    impact_probability: Optional[float] = Field(description="Impact probability")
    palermo_scale: Optional[float] = Field(description="Palermo scale value")
    torino_scale: Optional[int] = Field(description="Torino scale value")
    potential_impacts: Optional[int] = Field(description="Number of potential impacts")
    
    # Computed risk assessment
    risk_score: float = Field(description="Computed risk proxy score [0,1]")
    risk_bucket: RiskBucket = Field(description="Risk level category")
    risk_terms: RiskTerms = Field(description="Risk calculation components")
    score_notes: str = Field(description="Notes about score calculation")
    
    # Metadata
    is_pha: bool = Field(description="Is Potentially Hazardous Asteroid")
    data_sources: List[str] = Field(description="Data sources used")


class PaginationInfo(BaseModel):
    """Pagination metadata."""
    
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Results per page")
    total_rows: int = Field(description="Total number of results")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there are more pages")
    has_prev: bool = Field(description="Whether there are previous pages")


class PredictionResponse(BaseModel):
    """Response schema for NEO risk predictions."""
    
    predictions: List[NEOPrediction] = Field(description="List of NEO predictions")
    pagination: PaginationInfo = Field(description="Pagination information")
    
    # Response metadata
    generated_at: datetime = Field(description="When this response was generated")
    cache_hit: bool = Field(description="Whether result came from cache")
    etag: str = Field(description="ETag for caching")
    
    # Summary statistics
    summary: Dict[str, Any] = Field(
        description="Summary statistics about the results"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(description="Service health status")
    timestamp: datetime = Field(description="Health check timestamp")
    version: str = Field(description="API version")
    dependencies: Dict[str, str] = Field(
        description="Status of external dependencies"
    )


class MetricsResponse(BaseModel):
    """Metrics response for monitoring."""
    
    requests_total: int = Field(description="Total number of requests")
    requests_per_minute: float = Field(description="Current requests per minute")
    cache_hit_rate: float = Field(description="Cache hit rate percentage")
    average_response_time_ms: float = Field(description="Average response time")
    upstream_calls_total: int = Field(description="Total upstream API calls")
    error_rate: float = Field(description="Error rate percentage")
    active_connections: int = Field(description="Number of active connections")


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(description="Error type/code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(description="Additional error details")
    request_id: Optional[str] = Field(description="Request ID for tracking")
    timestamp: datetime = Field(description="Error timestamp")


# Type aliases for integration with other layers
QueryParams = Dict[str, Union[str, int, float, bool]]
ModelParams = Dict[str, Union[str, int, float]]
CacheKey = str