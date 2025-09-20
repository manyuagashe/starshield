"""
StarShield Prediction Layer - FastAPI Wrapper

This package provides a FastAPI-based wrapper for the StarShield prediction system,
orchestrating calls between Query and Model layers for NEO risk assessment.

Architecture:
- FastAPI application with async endpoints
- Pydantic schemas for request/response validation
- Abstract interfaces for Query and Model layer integration
- Redis/SQLite caching with ETag support
- Rate limiting and metrics collection
- Comprehensive error handling

Usage:
    from predictionlayer.main import app
    
    # Run with uvicorn
    # uvicorn predictionlayer.main:app --reload
"""

__version__ = "1.0.0"
__author__ = "StarShield Team"

from .main import app

__all__ = ["app"]