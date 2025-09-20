"""
Main prediction endpoints for NEO risk assessment.

This module handles the core API endpoints for generating risk predictions,
orchestrating calls between Query and Model layers, and managing caching.
"""

import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union
from math import ceil
import pandas as pd
import numpy as np

from fastapi import APIRouter, Depends, HTTPException, Header, Response, Request
from fastapi.responses import JSONResponse

from ..config import get_settings
from ..dependencies import (
    get_query_client_dep, 
    get_model_service_dep, 
    get_cache_client_dep,
    QueryClient,
    ModelService,
)
from ..cache import CacheClient


def safe_convert_value(value: Any, target_type: type, default: Any = None) -> Any:
    """Safely convert a pandas value to the target type."""
    if value is None or pd.isna(value):
        return default
    
    try:
        if target_type == float:
            if hasattr(value, 'item'):  # pandas scalar
                return float(value.item())
            return float(value)
        elif target_type == int:
            if hasattr(value, 'item'):  # pandas scalar
                return int(value.item())
            return int(value)
        elif target_type == str:
            return str(value)
        elif target_type == bool:
            return bool(value)
        else:
            return value
    except (ValueError, TypeError, AttributeError):
        return default
from ..schemas import (
    PredictionRequest, 
    PredictionResponse, 
    NEOPrediction,
    PaginationInfo,
    RiskTerms,
    RiskBucket,
)
from ..exceptions import ValidationError, UpstreamError, ModelError, DataNotFound
from ..middleware import get_metrics_middleware


router = APIRouter()


async def generate_predictions_data(
    request: PredictionRequest,
    query_client: QueryClient,
    model_service: ModelService,
) -> Dict[str, Any]:
    """
    Generate prediction data by orchestrating Query and Model layers.
    
    Args:
        request: Validated prediction request
        query_client: Query layer client
        model_service: Model layer service
        
    Returns:
        Dictionary with prediction results and metadata
    """
    # Record upstream calls for metrics
    metrics = get_metrics_middleware()
    
    try:
        # Step 1: Fetch data from Query layer
        if metrics:
            metrics.record_upstream_call()
        
        cad_data = await query_client.get_cad_data(
            date_min=request.date_min,
            date_max=request.date_max,
            dist_max_au=request.dist_max_au or 0.05,
            pha_only=request.pha_only,
        )
        
        if metrics:
            metrics.record_upstream_call()
        
        sentry_data = await query_client.get_sentry_data(
            date_min=request.date_min,
            date_max=request.date_max,
            ip_min=request.ip_min or 1e-8,
        )
        
        neows_data = None
        if request.include_neows:
            if metrics:
                metrics.record_upstream_call()
            neows_data = await query_client.get_neows_data(
                date_min=request.date_min,
                date_max=request.date_max,
            )
        
        # Step 2: Validate data quality
        data_quality = model_service.validate_data_quality(cad_data)
        
        # Step 3: Compute risk predictions
        predictions_df = await model_service.compute_risk_predictions(
            cad_data=cad_data,
            sentry_data=sentry_data,
            neows_data=neows_data,
            model_params=model_service.get_model_params(),
        )
        
        if predictions_df.empty:
            raise DataNotFound("No NEO data found for the specified criteria")
        
        # Step 4: Convert to API response format
        predictions = []
        for _, row in predictions_df.iterrows():
            # Handle datetime conversion
            close_approach_date = row.get('close_approach_date')
            if close_approach_date is not None and not pd.isna(close_approach_date):
                if hasattr(close_approach_date, 'to_pydatetime'):
                    close_approach_date = close_approach_date.to_pydatetime()
                elif isinstance(close_approach_date, str):
                    close_approach_date = datetime.fromisoformat(close_approach_date.replace('Z', '+00:00'))
            else:
                # Provide a default datetime for required field
                close_approach_date = datetime.now(timezone.utc)
            
            prediction = NEOPrediction(
                designation=safe_convert_value(row.get('designation'), str, ''),
                des_key=safe_convert_value(row.get('des_key'), str, ''),
                diameter_km=safe_convert_value(row.get('diameter_km'), float),
                absolute_magnitude=safe_convert_value(row.get('absolute_magnitude'), float),
                albedo=safe_convert_value(row.get('albedo'), float),
                close_approach_date=close_approach_date,
                distance_au=safe_convert_value(row.get('distance_au'), float, 0.0),
                velocity_km_s=safe_convert_value(row.get('velocity_km_s'), float, 0.0),
                impact_probability=safe_convert_value(row.get('impact_probability'), float),
                palermo_scale=safe_convert_value(row.get('palermo_scale'), float),
                torino_scale=safe_convert_value(row.get('torino_scale'), int),
                potential_impacts=safe_convert_value(row.get('potential_impacts'), int),
                risk_score=safe_convert_value(row.get('risk_score'), float, 0.0),
                risk_bucket=RiskBucket(safe_convert_value(row.get('risk_bucket'), str, 'low')),
                risk_terms=RiskTerms(
                    diameter_cubed=safe_convert_value(row.get('diameter_cubed_term'), float, 0.0),
                    velocity_squared=safe_convert_value(row.get('velocity_squared_term'), float, 0.0),
                    inverse_distance=safe_convert_value(row.get('inverse_distance_term'), float, 0.0),
                ),
                score_notes=safe_convert_value(row.get('score_notes'), str, ''),
                is_pha=safe_convert_value(row.get('is_pha'), bool, False),
                data_sources=row.get('data_sources', ['CAD']),
            )
            predictions.append(prediction)
        
        # Step 5: Apply pagination
        total_rows = len(predictions)
        start_idx = (request.page - 1) * request.page_size
        end_idx = start_idx + request.page_size
        paginated_predictions = predictions[start_idx:end_idx]
        
        # Step 6: Generate summary statistics
        risk_distribution = {}
        for bucket in RiskBucket:
            risk_distribution[bucket.value] = sum(
                1 for p in predictions if p.risk_bucket == bucket
            )
        
        summary = {
            "risk_distribution": risk_distribution,
            "date_range_days": (request.date_max - request.date_min).days,
            "total_close_approaches": total_rows,
            "data_quality": data_quality,
        }
        
        return {
            "predictions": paginated_predictions,
            "total_rows": total_rows,
            "summary": summary,
            "generated_at": datetime.utcnow(),
        }
        
    except Exception as e:
        if isinstance(e, (UpstreamError, ModelError, DataNotFound)):
            raise
        else:
            raise ModelError(f"Failed to generate predictions: {str(e)}")


@router.post("/", response_model=PredictionResponse)
async def get_predictions(
    request: PredictionRequest,
    query_client: QueryClient = Depends(get_query_client_dep),
    model_service: ModelService = Depends(get_model_service_dep),
    cache_client: CacheClient = Depends(get_cache_client_dep),
    if_none_match: Optional[str] = Header(None),
) -> Union[PredictionResponse, Response]:
    """
    Generate NEO risk predictions for the specified time range and criteria.
    
    This endpoint orchestrates data retrieval from NASA APIs via the Query layer,
    risk computation via the Model layer, and returns paginated results with caching.
    
    **Caching:**
    - Results are cached based on request parameters
    - ETags are supported for conditional requests (304 Not Modified)
    - Cache TTL is configurable (default: 6 hours)
    
    **Rate Limiting:**
    - 100 requests per minute per IP by default
    - Configurable via environment variables
    
    **Pagination:**
    - Use `page` and `page_size` parameters
    - Maximum page size is 1000 items
    """
    settings = get_settings()
    metrics = get_metrics_middleware()
    
    # Generate cache key from request parameters
    cache_key = cache_client.generate_cache_key(
        "predictions", 
        request.dict()
    )
    
    # Check cache first
    cached_result = await cache_client.get(cache_key)
    cache_hit = cached_result is not None
    
    if cache_hit and metrics:
        metrics.record_cache_hit()
    elif not cache_hit and metrics:
        metrics.record_cache_miss()
    
    if cached_result:
        # Check ETag for conditional request
        cached_etag = cached_result.get("etag", "")
        if if_none_match and cached_etag == if_none_match:
            return Response(status_code=304)
        
        # Return cached result
        pagination = PaginationInfo(
            page=request.page,
            page_size=request.page_size,
            total_rows=cached_result["total_rows"],
            total_pages=ceil(cached_result["total_rows"] / request.page_size),
            has_next=request.page * request.page_size < cached_result["total_rows"],
            has_prev=request.page > 1,
        )
        
        response = PredictionResponse(
            predictions=cached_result["predictions"],
            pagination=pagination,
            generated_at=datetime.fromisoformat(cached_result["generated_at"]),
            cache_hit=True,
            etag=cached_etag,
            summary=cached_result["summary"],
        )
        
        return response
    
    # Generate new predictions
    result_data = await generate_predictions_data(request, query_client, model_service)
    
    # Generate ETag
    etag = cache_client.generate_etag(result_data)
    
    # Check ETag for conditional request (even for new data)
    if if_none_match and etag == if_none_match:
        return Response(status_code=304)
    
    # Prepare response
    pagination = PaginationInfo(
        page=request.page,
        page_size=request.page_size,
        total_rows=result_data["total_rows"],
        total_pages=ceil(result_data["total_rows"] / request.page_size),
        has_next=request.page * request.page_size < result_data["total_rows"],
        has_prev=request.page > 1,
    )
    
    response = PredictionResponse(
        predictions=result_data["predictions"],
        pagination=pagination,
        generated_at=result_data["generated_at"],
        cache_hit=False,
        etag=etag,
        summary=result_data["summary"],
    )
    
    # Cache the result
    cache_data = {
        "predictions": [p.dict() for p in result_data["predictions"]],
        "total_rows": result_data["total_rows"],
        "summary": result_data["summary"],
        "generated_at": result_data["generated_at"].isoformat(),
        "etag": etag,
    }
    
    await cache_client.set(
        cache_key, 
        cache_data, 
        ttl=settings.cache_ttl_seconds
    )
    
    return response


@router.get("/summary")
async def get_summary(
    date_min: datetime,
    date_max: datetime,
    dist_max_au: float = 0.05,
    query_client: QueryClient = Depends(get_query_client_dep),
) -> Dict[str, Any]:
    """
    Get a summary of NEO activity for the specified date range.
    
    This endpoint provides a quick overview without full risk computation,
    useful for dashboard widgets and overview displays.
    """
    try:
        # Basic validation
        if date_max <= date_min:
            raise ValidationError("date_max must be after date_min")
        
        if (date_max - date_min).days > 365:
            raise ValidationError("Date range cannot exceed 365 days")
        
        # Get basic CAD data for summary
        cad_data = await query_client.get_cad_data(
            date_min=date_min,
            date_max=date_max,
            dist_max_au=dist_max_au,
        )
        
        if cad_data.empty:
            return {
                "date_range": {
                    "start": date_min.isoformat(),
                    "end": date_max.isoformat(),
                    "days": (date_max - date_min).days,
                },
                "total_objects": 0,
                "closest_approach": None,
                "fastest_object": None,
                "largest_object": None,
                "pha_count": 0,
            }
        
        # Calculate summary statistics
        closest_idx = cad_data['distance_au'].idxmin()
        fastest_idx = cad_data['velocity_km_s'].idxmax()
        largest_idx = cad_data['diameter_km'].idxmax() if 'diameter_km' in cad_data.columns else None
        
        summary = {
            "date_range": {
                "start": date_min.isoformat(),
                "end": date_max.isoformat(),
                "days": (date_max - date_min).days,
            },
            "total_objects": len(cad_data),
            "closest_approach": {
                "designation": str(cad_data.loc[closest_idx, 'designation']),
                "distance_au": safe_convert_value(cad_data.loc[closest_idx, 'distance_au'], float, 0.0),
                "date": str(cad_data.loc[closest_idx, 'close_approach_date']),
            },
            "fastest_object": {
                "designation": str(cad_data.loc[fastest_idx, 'designation']),
                "velocity_km_s": safe_convert_value(cad_data.loc[fastest_idx, 'velocity_km_s'], float, 0.0),
            },
            "pha_count": int(cad_data['is_pha'].sum()) if 'is_pha' in cad_data.columns else 0,
        }
        
        if largest_idx is not None:
            summary["largest_object"] = {
                "designation": str(cad_data.loc[largest_idx, 'designation']),
                "diameter_km": safe_convert_value(cad_data.loc[largest_idx, 'diameter_km'], float, 0.0),
            }
        else:
            summary["largest_object"] = None
        
        return summary
        
    except Exception as e:
        if isinstance(e, (ValidationError, UpstreamError)):
            raise
        else:
            raise UpstreamError(f"Failed to generate summary: {str(e)}", "query_layer")