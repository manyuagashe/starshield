"""
Custom middleware for the StarShield prediction layer.

This module provides rate limiting, metrics collection, and request tracking
middleware for the FastAPI application.
"""

import time
import uuid
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple, Optional
from datetime import datetime, timedelta

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings
from .exceptions import RateLimitExceeded


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm."""
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        # In-memory storage for rate limits (use Redis in production)
        self.clients: Dict[str, Deque[float]] = defaultdict(deque)
        
    def get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use X-Forwarded-For header if available (for proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        # Could also include API key or user ID here for per-user limits
        return client_ip
    
    def is_rate_limited(self, client_id: str) -> Tuple[bool, int]:
        """
        Check if client is rate limited.
        
        Returns:
            Tuple of (is_limited, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self.settings.rate_limit_window_seconds
        
        # Clean old requests outside the window
        client_requests = self.clients[client_id]
        while client_requests and client_requests[0] < window_start:
            client_requests.popleft()
        
        # Check if limit exceeded
        if len(client_requests) >= self.settings.rate_limit_requests:
            # Calculate retry after (when oldest request falls out of window)
            oldest_request = client_requests[0]
            retry_after = int(oldest_request + self.settings.rate_limit_window_seconds - now)
            return True, max(1, retry_after)
        
        # Add current request
        client_requests.append(now)
        return False, 0
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        client_id = self.get_client_id(request)
        
        is_limited, retry_after = self.is_rate_limited(client_id)
        if is_limited:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
        
        response = await call_next(request)
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting metrics and request tracking."""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "requests_total": 0,
            "requests_by_path": defaultdict(int),
            "requests_by_status": defaultdict(int),
            "response_times": deque(maxlen=1000),  # Keep last 1000 response times
            "errors_total": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "upstream_calls": 0,
        }
        self.start_time = time.time()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with metrics collection."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Record request start
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Update metrics
        self.metrics["requests_total"] += 1
        self.metrics["requests_by_path"][request.url.path] += 1
        self.metrics["requests_by_status"][response.status_code] += 1
        self.metrics["response_times"].append(response_time)
        
        if response.status_code >= 400:
            self.metrics["errors_total"] += 1
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"
        
        return response
    
    def get_metrics(self) -> Dict:
        """Get current metrics."""
        uptime = time.time() - self.start_time
        response_times = list(self.metrics["response_times"])
        
        return {
            "requests_total": self.metrics["requests_total"],
            "requests_per_minute": (
                self.metrics["requests_total"] / (uptime / 60) 
                if uptime > 0 else 0
            ),
            "average_response_time_ms": (
                sum(response_times) * 1000 / len(response_times)
                if response_times else 0
            ),
            "error_rate": (
                self.metrics["errors_total"] / self.metrics["requests_total"]
                if self.metrics["requests_total"] > 0 else 0
            ),
            "cache_hit_rate": (
                self.metrics["cache_hits"] / 
                (self.metrics["cache_hits"] + self.metrics["cache_misses"])
                if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0 
                else 0
            ),
            "upstream_calls_total": self.metrics["upstream_calls"],
            "uptime_seconds": uptime,
            "requests_by_path": dict(self.metrics["requests_by_path"]),
            "requests_by_status": dict(self.metrics["requests_by_status"]),
        }
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.metrics["cache_hits"] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.metrics["cache_misses"] += 1
    
    def record_upstream_call(self):
        """Record an upstream API call."""
        self.metrics["upstream_calls"] += 1


# Global metrics instance to be shared across the app
metrics_middleware_instance = None


def get_metrics_middleware() -> Optional[MetricsMiddleware]:
    """Get the global metrics middleware instance."""
    global metrics_middleware_instance
    return metrics_middleware_instance


def set_metrics_middleware(middleware: MetricsMiddleware):
    """Set the global metrics middleware instance."""
    global metrics_middleware_instance
    metrics_middleware_instance = middleware