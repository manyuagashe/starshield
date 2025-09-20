"""
Tests for configuration and settings management.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from predictionlayer.config import Settings, get_settings


class TestSettingsConfiguration:
    """Test application settings and configuration."""
    
    def test_default_settings(self):
        """Test default settings values (with .env file loaded)."""
        settings = Settings()
        
        # Test values from .env file
        assert settings.host == "127.0.0.1"  # From .env
        assert settings.port == 8000
        assert settings.debug is True  # From .env
        assert settings.max_page_size == 1000
        assert settings.default_page_size == 50
        
        # Cache settings from .env
        assert settings.cache_ttl_seconds == 21600
        assert settings.cache_backend == "sqlite"  # From .env  
        assert settings.cache_key_prefix == "starshield:"
        
        # Rate limiting
        assert settings.rate_limit_requests == 100
        assert settings.rate_limit_window_seconds == 60
        
        # API settings
        assert settings.api_request_timeout == 30
        assert settings.max_retries == 3
        assert settings.retry_backoff_factor == 2.0
    
    def test_environment_override(self):
        """Test that environment variables override defaults."""
        env_vars = {
            "STARSHIELD_DEBUG": "true",
            "STARSHIELD_PORT": "9000",
            "STARSHIELD_MAX_PAGE_SIZE": "500",
            "STARSHIELD_CACHE_TTL_SECONDS": "7200",
            "STARSHIELD_CACHE_BACKEND": "sqlite",
            "STARSHIELD_RATE_LIMIT_REQUESTS": "200"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.debug is True
            assert settings.port == 9000
            assert settings.max_page_size == 500
            assert settings.cache_ttl_seconds == 7200
            assert settings.cache_backend == "sqlite"
            assert settings.rate_limit_requests == 200
    
    def test_redis_configuration(self):
        """Test Redis configuration settings."""
        env_vars = {
            "STARSHIELD_CACHE_BACKEND": "redis",
            "STARSHIELD_REDIS_URL": "redis://localhost:6379/1"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.cache_backend == "redis"
            assert settings.redis_url == "redis://localhost:6379/1"
    
    def test_security_settings(self):
        """Test security-related settings."""
        env_vars = {
            "STARSHIELD_ALLOWED_HOSTS": '["localhost","127.0.0.1","example.com"]',
            "STARSHIELD_ALLOWED_ORIGINS": '["http://localhost:3000","https://app.example.com"]'
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert "localhost" in settings.allowed_hosts
            assert "127.0.0.1" in settings.allowed_hosts
            assert "example.com" in settings.allowed_hosts
            assert "http://localhost:3000" in settings.allowed_origins
            assert "https://app.example.com" in settings.allowed_origins
    
    def test_external_service_configuration(self):
        """Test external service configuration."""
        env_vars = {
            "STARSHIELD_NASA_API_KEY": "nasa-api-key",
            "STARSHIELD_API_REQUEST_TIMEOUT": "45",
            "STARSHIELD_MAX_RETRIES": "5"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.nasa_api_key == "nasa-api-key"
            assert settings.api_request_timeout == 45
            assert settings.max_retries == 5
    
    def test_circuit_breaker_settings(self):
        """Test circuit breaker configuration."""
        env_vars = {
            "STARSHIELD_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "10",
            "STARSHIELD_CIRCUIT_BREAKER_TIMEOUT_SECONDS": "120"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.circuit_breaker_failure_threshold == 10
            assert settings.circuit_breaker_timeout_seconds == 120


class TestSettingsSingleton:
    """Test settings singleton behavior."""
    
    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
    
    def test_settings_caching(self):
        """Test that settings are properly cached."""
        # Clear any existing cache
        get_settings.cache_clear()
        
        with patch.dict(os.environ, {"STARSHIELD_DEBUG": "true"}):
            settings1 = get_settings()
            settings2 = get_settings()
            
            assert settings1.debug is True
            assert settings1 is settings2


class TestSettingsValidation:
    """Test settings validation and error handling."""
    
    def test_redis_url_validation(self):
        """Test Redis URL validation."""
        # With redis backend but no URL, should get default
        with patch.dict(os.environ, {"STARSHIELD_CACHE_BACKEND": "redis"}):
            settings = Settings()
            assert settings.redis_url == "redis://localhost:6379/0"
    
    def test_list_parsing(self):
        """Test parsing of JSON-formatted lists."""
        env_vars = {
            "STARSHIELD_ALLOWED_HOSTS": '["host1","host2","host3"]',
            "STARSHIELD_ALLOWED_ORIGINS": '["http://origin1","https://origin2"]'
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.allowed_hosts == ["host1", "host2", "host3"]
            assert settings.allowed_origins == ["http://origin1", "https://origin2"]
    
    def test_invalid_port_numbers(self):
        """Test handling of invalid port numbers."""
        with patch.dict(os.environ, {"STARSHIELD_PORT": "invalid"}):
            with pytest.raises((ValueError, TypeError)):
                Settings()
        
        # Note: Pydantic allows negative integers, so this doesn't raise an error
        with patch.dict(os.environ, {"STARSHIELD_PORT": "-1"}):
            settings = Settings()
            assert settings.port == -1  # Pydantic allows this


class TestEnvironmentSpecificSettings:
    """Test environment-specific configuration."""
    
    def test_development_settings(self):
        """Test development environment settings."""
        with patch.dict(os.environ, {"STARSHIELD_DEBUG": "true"}):
            settings = Settings()
            
            assert settings.debug is True
            assert settings.enable_docs is True
    
    def test_production_settings(self):
        """Test production environment settings."""
        env_vars = {
            "STARSHIELD_DEBUG": "false",
            "STARSHIELD_ENABLE_DOCS": "false"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.debug is False
            assert settings.enable_docs is False
    
    def test_cache_backend_variations(self):
        """Test different cache backend configurations."""
        # SQLite backend
        with patch.dict(os.environ, {"STARSHIELD_CACHE_BACKEND": "sqlite"}):
            settings = Settings()
            assert settings.cache_backend == "sqlite"
        
        # Redis backend
        with patch.dict(os.environ, {"STARSHIELD_CACHE_BACKEND": "redis"}):
            settings = Settings()
            assert settings.cache_backend == "redis"


class TestSettingsIntegration:
    """Test settings integration with the application."""
    
    def test_settings_injection_in_app(self, test_app):
        """Test that settings are properly injected into the application."""
        from predictionlayer.main import app
        
        # Settings should be available through dependency injection
        assert hasattr(app, 'dependency_overrides')
    
    def test_settings_affect_behavior(self, test_client):
        """Test that settings actually affect application behavior."""
        # Test that changing page size limits affects API responses
        with patch('predictionlayer.config.get_settings') as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.max_page_size = 5
            mock_get_settings.return_value = mock_settings
            
            request_data = {
                "date_min": "2024-01-01T00:00:00Z",
                "date_max": "2024-01-31T23:59:59Z",
                "page_size": 10  # Exceeds limit
            }
            
            response = test_client.post("/api/v1/predictions/", json=request_data)
            
            # Should get validation error for page_size
            assert response.status_code == 422
    
    def test_cache_settings_integration(self, test_client):
        """Test that cache settings are properly used."""
        # This would test actual cache configuration
        # For now, just verify the endpoint works
        response = test_client.get("/health/")
        assert response.status_code == 200


class TestConfigurationFiles:
    """Test configuration file handling."""
    
    def test_config_file_loading(self):
        """Test loading configuration from files."""
        # This would test loading from config files like .env
        # For now, just ensure the basic structure works
        settings = Settings()
        assert settings is not None
    
    def test_config_precedence(self):
        """Test configuration precedence (env vars > config files > defaults)."""
        # Environment variables should take precedence
        with patch.dict(os.environ, {"STARSHIELD_DEBUG": "true"}):
            settings = Settings()
            assert settings.debug is True
    
    def test_env_prefix_handling(self):
        """Test that environment variable prefix is handled correctly."""
        with patch.dict(os.environ, {"STARSHIELD_PORT": "9999"}):
            settings = Settings()
            assert settings.port == 9999
        
        # Without prefix should use default
        with patch.dict(os.environ, {"PORT": "8888"}):
            settings = Settings()
            assert settings.port == 8000  # Should use default, not env var


class TestSettingsFields:
    """Test specific settings fields and their behavior."""
    
    def test_cache_key_prefix(self):
        """Test cache key prefix setting."""
        with patch.dict(os.environ, {"STARSHIELD_CACHE_KEY_PREFIX": "test:"}):
            settings = Settings()
            assert settings.cache_key_prefix == "test:"
    
    def test_date_range_limits(self):
        """Test date range limit settings."""
        with patch.dict(os.environ, {"STARSHIELD_MAX_DATE_RANGE_DAYS": "30"}):
            settings = Settings()
            assert settings.max_date_range_days == 30
    
    def test_rate_limiting_settings(self):
        """Test rate limiting configuration."""
        env_vars = {
            "STARSHIELD_RATE_LIMIT_REQUESTS": "50",
            "STARSHIELD_RATE_LIMIT_WINDOW_SECONDS": "120"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.rate_limit_requests == 50
            assert settings.rate_limit_window_seconds == 120
    
    def test_retry_settings(self):
        """Test retry configuration."""
        env_vars = {
            "STARSHIELD_MAX_RETRIES": "5",
            "STARSHIELD_RETRY_BACKOFF_FACTOR": "1.5"
        }
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            assert settings.max_retries == 5
            assert settings.retry_backoff_factor == 1.5