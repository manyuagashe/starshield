"""
Dependency injection and integration interfaces for other layers.

This module provides abstract interfaces and dependency injection setup
for integrating with the Query and Model layers built by other team members.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
from fastapi import Depends, Request

from .config import get_settings
from .schemas import QueryParams, ModelParams
from .exceptions import UpstreamError, ModelError


class QueryClient(ABC):
    """Abstract interface for the Query layer."""
    
    @abstractmethod
    async def get_cad_data(
        self, 
        date_min: datetime,
        date_max: datetime, 
        dist_max_au: float,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch CAD (Close Approach Data) from NASA APIs.
        
        Args:
            date_min: Start date for query
            date_max: End date for query  
            dist_max_au: Maximum distance in AU
            **kwargs: Additional query parameters
            
        Returns:
            DataFrame with normalized CAD records
            
        Raises:
            UpstreamError: When API calls fail
        """
        pass
    
    @abstractmethod
    async def get_sentry_data(
        self,
        date_min: datetime,
        date_max: datetime,
        ip_min: float,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch Sentry risk data from NASA APIs.
        
        Args:
            date_min: Start date for query
            date_max: End date for query
            ip_min: Minimum impact probability
            **kwargs: Additional query parameters
            
        Returns:
            DataFrame with Sentry risk records
            
        Raises:
            UpstreamError: When API calls fail
        """
        pass
    
    @abstractmethod
    async def get_neows_data(
        self,
        date_min: datetime,
        date_max: datetime,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Fetch NeoWs data for diameter estimates (optional).
        
        Args:
            date_min: Start date for query
            date_max: End date for query
            **kwargs: Additional query parameters
            
        Returns:
            DataFrame with NeoWs records or None if disabled
            
        Raises:
            UpstreamError: When API calls fail
        """
        pass


class ModelService(ABC):
    """Abstract interface for the Model layer."""
    
    @abstractmethod
    async def compute_risk_predictions(
        self,
        cad_data: pd.DataFrame,
        sentry_data: pd.DataFrame,
        neows_data: Optional[pd.DataFrame] = None,
        model_params: Optional[ModelParams] = None,
    ) -> pd.DataFrame:
        """
        Compute risk predictions from raw data.
        
        Args:
            cad_data: CAD DataFrame from Query layer
            sentry_data: Sentry DataFrame from Query layer
            neows_data: Optional NeoWs DataFrame
            model_params: Model configuration parameters
            
        Returns:
            DataFrame with computed risk scores and buckets
            
        Raises:
            ModelError: When computation fails
        """
        pass
    
    @abstractmethod
    def get_model_params(self) -> ModelParams:
        """Get current model parameters/configuration."""
        pass
    
    @abstractmethod
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return QA metrics.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            Dictionary with validation results and metrics
        """
        pass


class MockQueryClient(QueryClient):
    """Mock implementation for development/testing."""
    
    async def get_cad_data(
        self, 
        date_min: datetime,
        date_max: datetime, 
        dist_max_au: float,
        **kwargs
    ) -> pd.DataFrame:
        """Return mock CAD data."""
        # Create sample data structure that the real implementation would return
        import numpy as np
        
        n_objects = np.random.randint(10, 100)
        return pd.DataFrame({
            'des_key': [f'2024_AB{i:02d}' for i in range(n_objects)],
            'designation': [f'2024 AB{i:02d}' for i in range(n_objects)],
            'close_approach_date': pd.date_range(date_min, date_max, periods=n_objects),
            'distance_au': np.random.uniform(0.001, dist_max_au, n_objects),
            'velocity_km_s': np.random.uniform(5, 50, n_objects),
            'absolute_magnitude': np.random.uniform(15, 30, n_objects),
            'diameter_km': np.random.uniform(0.01, 10, n_objects),
            'is_pha': np.random.choice([True, False], n_objects, p=[0.1, 0.9]),
        })
    
    async def get_sentry_data(
        self,
        date_min: datetime,
        date_max: datetime,
        ip_min: float,
        **kwargs
    ) -> pd.DataFrame:
        """Return mock Sentry data."""
        import numpy as np
        
        n_objects = np.random.randint(5, 20)  # Fewer Sentry objects
        return pd.DataFrame({
            'des_key': [f'2024_AB{i:02d}' for i in range(n_objects)],
            'impact_probability': np.random.uniform(ip_min, 1e-3, n_objects),
            'palermo_scale': np.random.uniform(-5, 0, n_objects),
            'torino_scale': np.random.randint(0, 3, n_objects),
            'potential_impacts': np.random.randint(1, 100, n_objects),
        })
    
    async def get_neows_data(
        self,
        date_min: datetime,
        date_max: datetime,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """Return mock NeoWs data."""
        import numpy as np
        
        n_objects = np.random.randint(5, 30)
        return pd.DataFrame({
            'des_key': [f'2024_AB{i:02d}' for i in range(n_objects)],
            'diameter_km_mean': np.random.uniform(0.01, 5, n_objects),
            'albedo': np.random.uniform(0.05, 0.5, n_objects),
        })


class MockModelService(ModelService):
    """Mock implementation for development/testing."""
    
    def __init__(self):
        self.model_params = {
            'albedo_default': 0.14,
            'dist_penalty_eps': 0.001,
            'bucketing': 'tertiles',
            'risk_scaling': 'minmax',
        }
    
    async def compute_risk_predictions(
        self,
        cad_data: pd.DataFrame,
        sentry_data: pd.DataFrame,
        neows_data: Optional[pd.DataFrame] = None,
        model_params: Optional[ModelParams] = None,
    ) -> pd.DataFrame:
        """Return mock risk predictions."""
        import numpy as np
        
        # Simulate the model computation
        result_df = cad_data.copy()
        
        # Add risk computation columns
        result_df['risk_score'] = np.random.uniform(0, 1, len(result_df))
        result_df['risk_bucket'] = pd.cut(
            result_df['risk_score'], 
            bins=[0, 0.33, 0.67, 1.0], 
            labels=['low', 'medium', 'high']
        ).astype(str)
        
        # Add risk term breakdowns
        result_df['diameter_cubed_term'] = result_df['diameter_km'] ** 3
        result_df['velocity_squared_term'] = result_df['velocity_km_s'] ** 2
        result_df['inverse_distance_term'] = 1 / (result_df['distance_au'] + 0.001)
        
        result_df['score_notes'] = 'Mock calculation for development'
        result_df['data_sources'] = [['CAD', 'Mock'] for _ in range(len(result_df))]
        
        # Merge with Sentry data if available
        if not sentry_data.empty:
            result_df = result_df.merge(
                sentry_data, 
                on='des_key', 
                how='left'
            )
        
        return result_df
    
    def get_model_params(self) -> ModelParams:
        """Get current model parameters."""
        return self.model_params
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return mock data quality metrics."""
        return {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types_valid': True,
            'outliers_detected': 0,
            'quality_score': 0.95,
        }


# Dependency injection functions
def get_query_client() -> QueryClient:
    """Get Query client instance."""
    # TODO: Replace with real implementation when Query layer is ready
    return MockQueryClient()


def get_model_service() -> ModelService:
    """Get Model service instance.""" 
    # TODO: Replace with real implementation when Model layer is ready
    return MockModelService()


def get_cache_client_dep(request: Request):
    """FastAPI dependency for cache client."""
    return request.app.state.cache_client


def get_query_client_dep(request: Request) -> QueryClient:
    """FastAPI dependency for query client."""
    return request.app.state.query_client


def get_model_service_dep(request: Request) -> ModelService:
    """FastAPI dependency for model service."""
    return request.app.state.model_service