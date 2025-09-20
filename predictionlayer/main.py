"""
Main FastAPI application for the StarShield prediction layer wrapper.

This module provides the primary FastAPI app that orchestrates calls between
the Query and Model layers, handling validation, caching, and error management.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .middleware import RateLimitMiddleware, MetricsMiddleware
from .routers import health, metrics, predictions
from .exceptions import StarShieldException
from .cache import get_cache_client
from .dependencies import get_query_client, get_model_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup/shutdown events."""
    settings = get_settings()
    
    # Initialize cache client
    cache_client = get_cache_client()
    await cache_client.initialize()
    
    # Store in app state for dependency injection
    app.state.cache_client = cache_client
    app.state.query_client = get_query_client()
    app.state.model_service = get_model_service()
    
    yield
    
    # Cleanup
    await cache_client.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="StarShield Prediction API",
        description="Risk assessment and prediction system for near-Earth objects",
        version="1.0.0",
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        lifespan=lifespan,
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        max_age=3600,  # Cache preflight for 1 hour
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )
    
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(MetricsMiddleware)
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])
    app.include_router(
        predictions.router, 
        prefix="/api/v1/predictions", 
        tags=["Predictions"]
    )
    
    # Global exception handlers
    @app.exception_handler(StarShieldException)
    async def starshield_exception_handler(
        request: Request, exc: StarShieldException
    ) -> JSONResponse:
        """Handle custom StarShield exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_type,
                "message": exc.message,
                "details": exc.details,
                "request_id": getattr(request.state, "request_id", None),
            },
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None),
            },
        )
    
    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "predictionlayer.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )