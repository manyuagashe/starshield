#!/usr/bin/env python3
"""
Startup script for the StarShield prediction layer.

This script provides an easy way to run the FastAPI application
with proper configuration and error handling.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from predictionlayer.config import get_settings
    from predictionlayer.main import app
    import uvicorn
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install dependencies with: pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Run the FastAPI application."""
    settings = get_settings()
    
    print("🚀 Starting StarShield Prediction Layer")
    print(f"   Host: {settings.host}")
    print(f"   Port: {settings.port}")
    print(f"   Debug: {settings.debug}")
    print(f"   Cache: {settings.cache_backend}")
    print()
    print("📖 API Documentation:")
    print(f"   Swagger UI: http://{settings.host}:{settings.port}/docs")
    print(f"   ReDoc: http://{settings.host}:{settings.port}/redoc")
    print()
    print("🔍 Monitoring:")
    print(f"   Health: http://{settings.host}:{settings.port}/health/")
    print(f"   Metrics: http://{settings.host}:{settings.port}/metrics/")
    print()
    
    try:
        uvicorn.run(
            "predictionlayer.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.debug,
            log_level="debug" if settings.debug else "info",
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down StarShield Prediction Layer")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()