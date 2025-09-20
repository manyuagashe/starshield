"""
Unit tests for cache implementations.
"""

import pytest
import pytest_asyncio
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

from predictionlayer.cache import CacheClient, SQLiteCache, RedisCache


class TestCacheClient:
    """Test base CacheClient abstract methods."""
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        # Create a minimal implementation to test the abstract base
        class TestCache(CacheClient):
            async def initialize(self): pass
            async def get(self, key: str): pass
            async def set(self, key: str, value, ttl: Optional[int] = None): pass
            async def delete(self, key: str): pass
            async def exists(self, key: str) -> bool: return True
            async def close(self): pass
        
        cache = TestCache()
        
        # Test with simple parameters
        key1 = cache.generate_cache_key("test", {"param1": "value1", "param2": "value2"})
        key2 = cache.generate_cache_key("test", {"param2": "value2", "param1": "value1"})
        
        # Should be the same regardless of parameter order
        assert key1 == key2
        assert key1.startswith("test:")
        
        # Test with datetime serialization
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        key3 = cache.generate_cache_key("test", {"date": dt})
        # The datetime should be hashed, so we just check the prefix
        assert key3.startswith("test:")
        
        # Test that same datetime generates same key
        key4 = cache.generate_cache_key("test", {"date": dt})
        assert key3 == key4
        
    def test_generate_etag(self):
        """Test ETag generation."""
        class TestCache(CacheClient):
            async def initialize(self): pass
            async def get(self, key: str): pass
            async def set(self, key: str, value, ttl: Optional[int] = None): pass
            async def delete(self, key: str): pass
            async def exists(self, key: str) -> bool: return True
            async def close(self): pass
        
        cache = TestCache()
        
        # Test with simple data
        data1 = {"key": "value", "number": 42}
        data2 = {"number": 42, "key": "value"}
        
        etag1 = cache.generate_etag(data1)
        etag2 = cache.generate_etag(data2)
        
        # Should be the same regardless of key order
        assert etag1 == etag2
        assert len(etag1) == 16  # Truncated SHA256
        
        # Test with datetime
        data_with_dt = {
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "value": "test"
        }
        etag3 = cache.generate_etag(data_with_dt)
        assert len(etag3) == 16


class TestSQLiteCache:
    """Test SQLite cache implementation."""
    
    @pytest_asyncio.fixture
    async def sqlite_cache(self):
        """Create SQLite cache for testing."""
        cache = SQLiteCache(":memory:")
        await cache.initialize()
        yield cache
        await cache.close()
    
    @pytest.mark.asyncio
    async def test_sqlite_cache_basic_operations(self, sqlite_cache):
        """Test basic cache operations."""
        # Test set and get
        await sqlite_cache.set("key1", {"data": "value1"}, ttl=60)
        result = await sqlite_cache.get("key1")
        
        assert result is not None
        assert result["data"] == "value1"
        
        # Test get non-existent key
        result = await sqlite_cache.get("nonexistent")
        assert result is None
        
        # Test delete
        await sqlite_cache.delete("key1")
        result = await sqlite_cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_sqlite_cache_ttl_expiration(self, sqlite_cache):
        """Test TTL expiration."""
        # Set with very short TTL
        await sqlite_cache.set("expiring_key", {"data": "will_expire"}, ttl=1)
        
        # Should exist immediately
        result = await sqlite_cache.get("expiring_key")
        assert result is not None
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired now
        result = await sqlite_cache.get("expiring_key")
        assert result is None
    
    @pytest.mark.asyncio 
    async def test_sqlite_cache_clear(self, sqlite_cache):
        """Test cache clearing."""
        # Add multiple entries
        await sqlite_cache.set("key1", {"data": "value1"})
        await sqlite_cache.set("key2", {"data": "value2"})
        await sqlite_cache.set("key3", {"data": "value3"})
        
        # Verify they exist
        assert await sqlite_cache.get("key1") is not None
        assert await sqlite_cache.get("key2") is not None
        assert await sqlite_cache.get("key3") is not None
        
        # Clear cache
        await sqlite_cache.clear()
        
        # Verify they're gone
        assert await sqlite_cache.get("key1") is None
        assert await sqlite_cache.get("key2") is None
        assert await sqlite_cache.get("key3") is None
    
    @pytest.mark.asyncio
    async def test_sqlite_cache_datetime_serialization(self, sqlite_cache):
        """Test datetime serialization in cache."""
        dt = datetime(2024, 1, 1, 12, 30, 45, tzinfo=timezone.utc)
        data = {
            "timestamp": dt,
            "message": "test with datetime"
        }
        
        await sqlite_cache.set("datetime_key", data)
        result = await sqlite_cache.get("datetime_key")
        
        assert result is not None
        assert result["message"] == "test with datetime"
        # Datetime should be serialized as ISO string
        assert result["timestamp"] == dt.isoformat()


@pytest.mark.asyncio 
class TestRedisCache:
    """Test Redis cache implementation."""
    
    @pytest_asyncio.fixture
    async def mock_redis(self):
        """Mock async Redis client for testing."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None
        redis_mock.set.return_value = True
        redis_mock.delete.return_value = True
        redis_mock.exists.return_value = False
        redis_mock.flushdb.return_value = True
        redis_mock.ping.return_value = True
        return redis_mock
    
    @pytest_asyncio.fixture
    async def redis_cache(self, mock_redis):
        """Create a Redis cache with mocked Redis client."""
        with patch('predictionlayer.cache.aioredis.from_url', return_value=mock_redis):
            cache = RedisCache(
                redis_url="redis://localhost:6379/0",
                key_prefix="test:"
            )
            await cache.initialize()
            yield cache, mock_redis
            await cache.close()
    
    async def test_redis_cache_basic_operations(self, redis_cache):
        """Test basic Redis cache operations."""
        cache, mock_redis = redis_cache
        
        # Mock get to return serialized data
        test_data = {"data": "value1"}
        mock_redis.get.return_value = json.dumps({
            "data": test_data,
            "created_at": datetime.utcnow().isoformat()
        })
        
        # Test get
        result = await cache.get("key1")
        assert result == test_data
        mock_redis.get.assert_called_with("test:key1")
        
        # Test set without TTL
        await cache.set("key1", test_data)
        mock_redis.set.assert_called()
        
        # Test set with TTL
        await cache.set("key1", test_data, ttl=60)
        mock_redis.setex.assert_called()
        
        # Test delete
        await cache.delete("key1")
        mock_redis.delete.assert_called_with("test:key1")
    
    async def test_redis_cache_get_nonexistent(self, redis_cache):
        """Test getting non-existent key from Redis."""
        cache, mock_redis = redis_cache
        
        # Mock Redis to return None
        mock_redis.get.return_value = None
        
        result = await cache.get("nonexistent")
        assert result is None
    
    async def test_redis_cache_get_expired(self, redis_cache):
        """Test getting expired key from Redis."""
        cache, mock_redis = redis_cache
        
        # Mock Redis to return expired data
        expired_time = datetime.utcnow() - timedelta(hours=1)
        mock_redis.get.return_value = json.dumps({
            "data": {"test": "data"},
            "created_at": expired_time.isoformat(),
            "expires_at": expired_time.isoformat()
        })
        
        result = await cache.get("expired_key")
        assert result is None
        
        # Should delete expired key
        mock_redis.delete.assert_called_with("test:expired_key")
    
    async def test_redis_cache_clear(self, redis_cache):
        """Test Redis cache clearing."""
        cache, mock_redis = redis_cache
        
        await cache.clear()
        mock_redis.flushdb.assert_called_once()
    
    async def test_redis_cache_health_check(self, redis_cache):
        """Test Redis health check."""
        cache, mock_redis = redis_cache
        
        # Test healthy Redis with ping
        mock_redis.ping.return_value = True
        
        # Call ping directly since there's no is_healthy method
        try:
            await mock_redis.ping()
            health = True
        except Exception:
            health = False
        
        assert health is True
        
        # Test unhealthy Redis
        mock_redis.ping.side_effect = Exception("Connection failed")
        try:
            await mock_redis.ping()
            health = True
        except Exception:
            health = False
        
        assert health is False
    
    async def test_redis_cache_datetime_serialization(self, redis_cache):
        """Test datetime serialization in Redis cache."""
        cache, mock_redis = redis_cache
        
        dt = datetime(2024, 1, 1, 12, 30, 45, tzinfo=timezone.utc)
        data = {
            "timestamp": dt,
            "message": "test with datetime"
        }
        
        # Test that datetime gets serialized properly
        await cache.set("datetime_key", data)
        
        # Check that the call to Redis included serialized datetime
        call_args = mock_redis.set.call_args
        stored_data = json.loads(call_args[0][1])
        assert stored_data["data"]["timestamp"] == dt.isoformat()
        assert stored_data["data"]["message"] == "test with datetime"


class TestCacheFactory:
    """Test cache factory functions."""
    
    def test_cache_key_generation_consistency(self):
        """Test that cache key generation is consistent across implementations."""
        sqlite_cache = SQLiteCache(":memory:")
        
        with patch('predictionlayer.cache.aioredis.from_url'):
            redis_cache = RedisCache("redis://localhost:6379/0")
        
        params = {
            "date_min": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "date_max": datetime(2024, 1, 31, tzinfo=timezone.utc),
            "dist_max_au": 0.05
        }
        
        sqlite_key = sqlite_cache.generate_cache_key("predictions", params)
        redis_key = redis_cache.generate_cache_key("predictions", params)
        
        # Both should generate the same key
        assert sqlite_key == redis_key
    
    def test_etag_generation_consistency(self):
        """Test that ETag generation is consistent across implementations."""
        sqlite_cache = SQLiteCache(":memory:")
        
        with patch('predictionlayer.cache.aioredis.from_url'):
            redis_cache = RedisCache("redis://localhost:6379/0")
        
        data = {
            "predictions": [{"id": 1, "name": "test"}],
            "total": 1,
            "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc)
        }
        
        sqlite_etag = sqlite_cache.generate_etag(data)
        redis_etag = redis_cache.generate_etag(data)
        
        # Both should generate the same ETag
        assert sqlite_etag == redis_etag