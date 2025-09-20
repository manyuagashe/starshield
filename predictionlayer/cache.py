"""
Caching layer for the StarShield prediction layer.

This module provides caching implementations with support for Redis and SQLite,
ETags, TTL management, and cache invalidation strategies.
"""

import asyncio
import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import aioredis

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

from .config import get_settings
from .exceptions import CacheError


class CacheClient(ABC):
    """Abstract base class for cache implementations."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the cache client."""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a value from cache."""
        pass
    
    @abstractmethod
    async def set(
        self, 
        key: str, 
        value: Dict[str, Any], 
        ttl: Optional[int] = None
    ) -> None:
        """Set a value in cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a key from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the cache client."""
        pass
    
    def generate_cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate a deterministic cache key from parameters."""
        # Convert non-serializable objects to strings
        serializable_params = {}
        for key, value in params.items():
            if hasattr(value, 'isoformat'):  # datetime objects
                serializable_params[key] = value.isoformat()
            else:
                serializable_params[key] = value
        
        # Sort parameters for consistent key generation
        sorted_params = json.dumps(serializable_params, sort_keys=True, separators=(",", ":"))
        param_hash = hashlib.sha256(sorted_params.encode()).hexdigest()[:16]
        return f"{prefix}:{param_hash}"
    
    def generate_etag(self, data: Any) -> str:
        """Generate ETag for response data."""
        def serialize_datetime(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            if hasattr(obj, 'dict'):  # Pydantic models
                return obj.dict()
            raise TypeError(f"Object of type {type(obj).__name__} is not serializable")
        
        data_str = json.dumps(data, sort_keys=True, separators=(",", ":"), default=serialize_datetime)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]


class RedisCache(CacheClient):
    """Redis-based cache implementation."""
    
    def __init__(self, redis_url: str, key_prefix: str = "starshield:"):
        if not REDIS_AVAILABLE or aioredis is None:
            raise CacheError("Redis is not available. Install with: pip install aioredis")
        
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis: Optional[Any] = None
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            if aioredis is None:
                raise CacheError("aioredis module not available")
            self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            await self.redis.ping()
        except Exception as e:
            raise CacheError(f"Failed to connect to Redis: {e}")
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from Redis."""
        if not self.redis:
            raise CacheError("Redis client not initialized")
        
        try:
            full_key = f"{self.key_prefix}{key}"
            data = await self.redis.get(full_key)
            if data:
                cached_item = json.loads(data)
                
                # Check if item has expired
                if "expires_at" in cached_item:
                    expires_at = datetime.fromisoformat(cached_item["expires_at"])
                    if datetime.utcnow() > expires_at:
                        await self.delete(key)
                        return None
                
                return cached_item["data"]
            return None
        except json.JSONDecodeError:
            # Invalid cached data, remove it
            await self.delete(key)
            return None
        except Exception as e:
            raise CacheError(f"Failed to get from Redis cache: {e}")
    
    async def set(
        self, 
        key: str, 
        value: Dict[str, Any], 
        ttl: Optional[int] = None
    ) -> None:
        """Set value in Redis."""
        if not self.redis:
            raise CacheError("Redis client not initialized")
        
        try:
            full_key = f"{self.key_prefix}{key}"
            
            # Prepare cache item with metadata
            cache_item = {
                "data": value,
                "created_at": datetime.utcnow().isoformat(),
            }
            
            if ttl:
                cache_item["expires_at"] = (
                    datetime.utcnow() + timedelta(seconds=ttl)
                ).isoformat()
            
            def serialize_datetime(obj):
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                if hasattr(obj, 'dict'):  # Pydantic models
                    return obj.dict()
                raise TypeError(f"Object of type {type(obj).__name__} is not serializable")
            
            data = json.dumps(cache_item, separators=(",", ":"), default=serialize_datetime)
            
            if ttl:
                await self.redis.setex(full_key, ttl, data)
            else:
                await self.redis.set(full_key, data)
                
        except Exception as e:
            raise CacheError(f"Failed to set Redis cache: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete key from Redis."""
        if not self.redis:
            raise CacheError("Redis client not initialized")
        
        try:
            full_key = f"{self.key_prefix}{key}"
            await self.redis.delete(full_key)
        except Exception as e:
            raise CacheError(f"Failed to delete from Redis cache: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self.redis:
            raise CacheError("Redis client not initialized")
        
        try:
            full_key = f"{self.key_prefix}{key}"
            return bool(await self.redis.exists(full_key))
        except Exception as e:
            raise CacheError(f"Failed to check Redis cache existence: {e}")
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()


class SQLiteCache(CacheClient):
    """SQLite-based cache implementation for development/testing."""
    
    def __init__(self, db_path: str = ":memory:", key_prefix: str = "starshield:"):
        self.db_path = db_path
        self.key_prefix = key_prefix
        self.connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                )
            """)
            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON cache(expires_at)
            """)
            self.connection.commit()
        except Exception as e:
            raise CacheError(f"Failed to initialize SQLite cache: {e}")
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from SQLite."""
        if not self.connection:
            raise CacheError("SQLite cache not initialized")
        
        async with self._lock:
            try:
                full_key = f"{self.key_prefix}{key}"
                cursor = self.connection.execute(
                    "SELECT data, expires_at FROM cache WHERE key = ?",
                    (full_key,)
                )
                row = cursor.fetchone()
                
                if row:
                    data_str, expires_at = row
                    
                    # Check expiration
                    if expires_at:
                        expires_at_dt = datetime.fromisoformat(expires_at)
                        if datetime.utcnow() > expires_at_dt:
                            await self._delete_sync(full_key)
                            return None
                    
                    return json.loads(data_str)
                return None
            except json.JSONDecodeError:
                # Invalid cached data, remove it
                await self._delete_sync(full_key)
                return None
            except Exception as e:
                raise CacheError(f"Failed to get from SQLite cache: {e}")
    
    async def set(
        self, 
        key: str, 
        value: Dict[str, Any], 
        ttl: Optional[int] = None
    ) -> None:
        """Set value in SQLite."""
        if not self.connection:
            raise CacheError("SQLite cache not initialized")
        
        async with self._lock:
            try:
                full_key = f"{self.key_prefix}{key}"
                
                def serialize_datetime(obj):
                    if hasattr(obj, 'isoformat'):
                        return obj.isoformat()
                    if hasattr(obj, 'dict'):  # Pydantic models
                        return obj.dict()
                    raise TypeError(f"Object of type {type(obj).__name__} is not serializable")
                
                data_str = json.dumps(value, separators=(",", ":"), default=serialize_datetime)
                created_at = datetime.utcnow().isoformat()
                
                expires_at = None
                if ttl:
                    expires_at = (
                        datetime.utcnow() + timedelta(seconds=ttl)
                    ).isoformat()
                
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO cache 
                    (key, data, created_at, expires_at) 
                    VALUES (?, ?, ?, ?)
                    """,
                    (full_key, data_str, created_at, expires_at)
                )
                self.connection.commit()
            except Exception as e:
                raise CacheError(f"Failed to set SQLite cache: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete key from SQLite."""
        if not self.connection:
            raise CacheError("SQLite cache not initialized")
        
        async with self._lock:
            await self._delete_sync(f"{self.key_prefix}{key}")
    
    async def _delete_sync(self, full_key: str) -> None:
        """Synchronous delete helper."""
        try:
            if self.connection:
                self.connection.execute("DELETE FROM cache WHERE key = ?", (full_key,))
                self.connection.commit()
        except Exception as e:
            raise CacheError(f"Failed to delete from SQLite cache: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in SQLite."""
        if not self.connection:
            raise CacheError("SQLite cache not initialized")
        
        async with self._lock:
            try:
                full_key = f"{self.key_prefix}{key}"
                cursor = self.connection.execute(
                    "SELECT 1 FROM cache WHERE key = ? LIMIT 1",
                    (full_key,)
                )
                return cursor.fetchone() is not None
            except Exception as e:
                raise CacheError(f"Failed to check SQLite cache existence: {e}")
    
    async def close(self) -> None:
        """Close SQLite connection."""
        if self.connection:
            self.connection.close()


def get_cache_client() -> CacheClient:
    """Factory function to get the appropriate cache client."""
    settings = get_settings()
    
    if settings.cache_backend == "redis":
        if not REDIS_AVAILABLE:
            raise CacheError(
                "Redis backend requested but aioredis is not installed. "
                "Install with: pip install aioredis"
            )
        redis_url = settings.redis_url or "redis://localhost:6379/0"
        return RedisCache(
            redis_url=redis_url,
            key_prefix=settings.cache_key_prefix,
        )
    elif settings.cache_backend == "sqlite":
        return SQLiteCache(
            db_path="cache.db",
            key_prefix=settings.cache_key_prefix,
        )
    else:
        raise CacheError(f"Unsupported cache backend: {settings.cache_backend}")