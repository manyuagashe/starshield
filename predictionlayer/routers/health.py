"""
Health check endpoints for monitoring service status.
"""

from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends

from ..dependencies import get_query_client_dep, get_model_service_dep, get_cache_client_dep
from ..schemas import HealthResponse


router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check(
    query_client=Depends(get_query_client_dep),
    model_service=Depends(get_model_service_dep),
    cache_client=Depends(get_cache_client_dep),
) -> HealthResponse:
    """
    Comprehensive health check for all service dependencies.
    
    This endpoint checks the health of:
    - Cache layer (Redis/SQLite)
    - Query layer (NASA APIs)
    - Model layer (computation service)
    """
    dependencies = {}
    
    # Check cache health
    try:
        await cache_client.exists("health_check")
        dependencies["cache"] = "healthy"
    except Exception:
        dependencies["cache"] = "unhealthy"
    
    # Check query client (basic validation)
    try:
        # Just check if the client is instantiated properly
        query_client.get_model_params()  # This should work for mock
        dependencies["query_layer"] = "healthy"
    except Exception:
        dependencies["query_layer"] = "unhealthy"
    
    # Check model service
    try:
        model_service.get_model_params()
        dependencies["model_layer"] = "healthy"
    except Exception:
        dependencies["model_layer"] = "unhealthy"
    
    # Determine overall status
    all_healthy = all(status == "healthy" for status in dependencies.values())
    overall_status = "healthy" if all_healthy else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="1.0.0",
        dependencies=dependencies,
    )


@router.get("/liveness")
async def liveness_check() -> Dict[str, str]:
    """
    Simple liveness check for container orchestration.
    
    Returns 200 OK if the service is running.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/readiness")
async def readiness_check(
    cache_client=Depends(get_cache_client_dep),
) -> Dict[str, str]:
    """
    Readiness check for container orchestration.
    
    Returns 200 OK if the service is ready to handle requests.
    """
    try:
        # Basic check that critical dependencies are available
        await cache_client.exists("readiness_check")
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}