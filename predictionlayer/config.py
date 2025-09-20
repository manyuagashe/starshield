"""
Configuration management for the StarShield prediction layer.

This module handles environment variables, settings validation, 
and provides a centralized configuration interface.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    enable_docs: bool = Field(default=True, description="Enable API documentation")
    
    # Security settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000"], 
        description="Allowed CORS origins"
    )
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1"], 
        description="Allowed trusted hosts"
    )
    
    # Cache settings
    cache_backend: str = Field(default="redis", description="Cache backend (redis/sqlite)")
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    cache_ttl_seconds: int = Field(default=21600, description="Cache TTL (6 hours)")
    cache_key_prefix: str = Field(default="starshield:", description="Cache key prefix")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Requests per minute per IP")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window")
    
    # External API settings
    nasa_api_key: Optional[str] = Field(default=None, description="NASA API key")
    api_request_timeout: int = Field(default=30, description="API request timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_backoff_factor: float = Field(default=2.0, description="Retry backoff factor")
    
    # Data processing settings
    max_date_range_days: int = Field(default=365, description="Maximum date range in days")
    default_page_size: int = Field(default=50, description="Default pagination size")
    max_page_size: int = Field(default=1000, description="Maximum pagination size")
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = Field(
        default=5, 
        description="Circuit breaker failure threshold"
    )
    circuit_breaker_timeout_seconds: int = Field(
        default=60, 
        description="Circuit breaker timeout"
    )
    
    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v):
        """Parse comma-separated origins from environment variable."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("allowed_hosts", mode="before") 
    @classmethod
    def parse_hosts(cls, v):
        """Parse comma-separated hosts from environment variable."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v, info):
        """Validate Redis URL when using Redis cache backend."""
        # We need to get cache_backend from the values
        if hasattr(info, 'data') and info.data.get("cache_backend", "redis") == "redis" and not v:
            return "redis://localhost:6379/0"
        return v
    
    class Config:
        env_file = ".env"
        env_prefix = "STARSHIELD_"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()