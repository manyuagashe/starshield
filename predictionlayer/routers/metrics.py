"""
Metrics endpoints for monitoring and observability.
"""

from fastapi import APIRouter, Request

from ..schemas import MetricsResponse


router = APIRouter()


@router.get("/", response_model=MetricsResponse)
async def get_metrics(request: Request) -> MetricsResponse:
    """
    Get application metrics for monitoring.
    
    Returns performance metrics, cache statistics, and error rates
    that can be used by monitoring systems like Prometheus.
    """
    # Get metrics from middleware
    from ..middleware import get_metrics_middleware
    
    metrics_middleware = get_metrics_middleware()
    if not metrics_middleware:
        # Fallback if middleware not available
        return MetricsResponse(
            requests_total=0,
            requests_per_minute=0.0,
            cache_hit_rate=0.0,
            average_response_time_ms=0.0,
            upstream_calls_total=0,
            error_rate=0.0,
            active_connections=0,
        )
    
    metrics_data = metrics_middleware.get_metrics()
    
    return MetricsResponse(
        requests_total=metrics_data["requests_total"],
        requests_per_minute=metrics_data["requests_per_minute"],
        cache_hit_rate=metrics_data["cache_hit_rate"] * 100,  # Convert to percentage
        average_response_time_ms=metrics_data["average_response_time_ms"],
        upstream_calls_total=metrics_data["upstream_calls_total"],
        error_rate=metrics_data["error_rate"] * 100,  # Convert to percentage
        active_connections=0,  # TODO: Implement connection tracking
    )


@router.get("/prometheus")
async def prometheus_metrics(request: Request) -> str:
    """
    Export metrics in Prometheus format.
    
    This endpoint provides metrics in the format expected by Prometheus
    for easy integration with monitoring infrastructure.
    """
    from ..middleware import get_metrics_middleware
    
    metrics_middleware = get_metrics_middleware()
    if not metrics_middleware:
        return "# No metrics available\n"
    
    metrics_data = metrics_middleware.get_metrics()
    
    # Format metrics in Prometheus exposition format
    prometheus_output = []
    
    # Add help and type information
    prometheus_output.extend([
        "# HELP starshield_requests_total Total number of HTTP requests",
        "# TYPE starshield_requests_total counter",
        f"starshield_requests_total {metrics_data['requests_total']}",
        "",
        "# HELP starshield_request_duration_seconds Request duration in seconds",
        "# TYPE starshield_request_duration_seconds histogram",
        f"starshield_request_duration_seconds_sum {sum(metrics_data.get('response_times', []))}",
        f"starshield_request_duration_seconds_count {len(metrics_data.get('response_times', []))}",
        "",
        "# HELP starshield_cache_hit_rate Cache hit rate as a ratio",
        "# TYPE starshield_cache_hit_rate gauge",
        f"starshield_cache_hit_rate {metrics_data['cache_hit_rate']}",
        "",
        "# HELP starshield_upstream_calls_total Total upstream API calls",
        "# TYPE starshield_upstream_calls_total counter",
        f"starshield_upstream_calls_total {metrics_data['upstream_calls_total']}",
        "",
        "# HELP starshield_error_rate Error rate as a ratio",
        "# TYPE starshield_error_rate gauge",
        f"starshield_error_rate {metrics_data['error_rate']}",
        "",
    ])
    
    # Add per-path metrics
    prometheus_output.extend([
        "# HELP starshield_requests_by_path Requests grouped by path",
        "# TYPE starshield_requests_by_path counter",
    ])
    
    for path, count in metrics_data.get("requests_by_path", {}).items():
        safe_path = path.replace('"', '\\"')
        prometheus_output.append(f'starshield_requests_by_path{{path="{safe_path}"}} {count}')
    
    prometheus_output.append("")
    
    # Add per-status metrics  
    prometheus_output.extend([
        "# HELP starshield_requests_by_status Requests grouped by status code",
        "# TYPE starshield_requests_by_status counter",
    ])
    
    for status, count in metrics_data.get("requests_by_status", {}).items():
        prometheus_output.append(f'starshield_requests_by_status{{status="{status}"}} {count}')
    
    return "\n".join(prometheus_output)