import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.memory.base import BaseMemory
from hypertic.memory.redis.cache import RedisCache


class TestRedisCache:
    @pytest.fixture
    def mock_store(self):
        store = Mock(spec=BaseMemory)
        store.get_messages = MagicMock(return_value=[])
        store.save_message = MagicMock()
        store.setup = MagicMock()
        return store

    @pytest.fixture
    def redis_cache(self, mock_store):
        with patch("hypertic.memory.redis.cache.redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping = MagicMock()
            mock_client.get = MagicMock(return_value=None)
            mock_client.setex = MagicMock()
            mock_client.delete = MagicMock()
            mock_redis.from_url.return_value = mock_client
            return RedisCache(store=mock_store, redis_url="redis://localhost:6379")

    def test_redis_cache_creation(self, redis_cache, mock_store):
        """Test RedisCache initialization."""
        assert redis_cache.store == mock_store
        assert redis_cache.ttl == 3600
        assert redis_cache.key_prefix == "cache:memory:"

    def test_redis_cache_creation_custom_params(self, mock_store):
        """Test RedisCache with custom parameters."""
        with patch("hypertic.memory.redis.cache.redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping = MagicMock()
            mock_redis.from_url.return_value = mock_client
            cache = RedisCache(store=mock_store, redis_url="redis://localhost:6379", ttl=7200, key_prefix="custom:")
            assert cache.ttl == 7200
            assert cache.key_prefix == "custom:"

    def test_make_cache_key_with_session_id(self, redis_cache):
        """Test _make_cache_key with session_id."""
        key = redis_cache._make_cache_key(session_id="session1")
        assert key is not None
        assert key.startswith("cache:memory:")

    def test_make_cache_key_with_user_id(self, redis_cache):
        """Test _make_cache_key with user_id."""
        key = redis_cache._make_cache_key(user_id="user1")
        assert key is not None
        assert key.startswith("cache:memory:")

    def test_make_cache_key_with_both(self, redis_cache):
        """Test _make_cache_key with both session_id and user_id."""
        key = redis_cache._make_cache_key(session_id="session1", user_id="user1")
        assert key is not None
        assert key.startswith("cache:memory:")

    def test_make_cache_key_with_none(self, redis_cache):
        """Test _make_cache_key with no parameters."""
        key = redis_cache._make_cache_key()
        assert key is None

    def test_get_messages_cache_miss(self, redis_cache, mock_store):
        """Test get_messages with cache miss."""
        mock_store.get_messages.return_value = [{"role": "user", "content": "Hello"}]
        redis_cache.redis_client.get.return_value = None

        result = redis_cache.get_messages(session_id="session1", user_id="user1")
        assert len(result) == 1
        mock_store.get_messages.assert_called_once()
        redis_cache.redis_client.setex.assert_called_once()

    def test_get_messages_cache_hit(self, redis_cache, mock_store):
        """Test get_messages with cache hit."""
        cached_data = json.dumps([{"role": "user", "content": "Hello"}])
        redis_cache.redis_client.get.return_value = cached_data

        result = redis_cache.get_messages(session_id="session1", user_id="user1")
        assert len(result) == 1
        mock_store.get_messages.assert_not_called()

    def test_save_message(self, redis_cache, mock_store):
        """Test save_message."""
        mock_store.get_messages.return_value = [{"role": "user", "content": "Hello"}]
        redis_cache.save_message("session1", "user", "Hello", user_id="user1")

        mock_store.save_message.assert_called_once()
        # After saving, it updates the cache with new messages
        redis_cache.redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_messages_cache_miss(self, redis_cache, mock_store):
        """Test aget_messages with cache miss."""
        mock_store.aget_messages = AsyncMock(return_value=[{"role": "user", "content": "Hello"}])
        redis_cache.redis_client.get = MagicMock(return_value=None)

        result = await redis_cache.aget_messages(session_id="session1", user_id="user1")
        assert len(result) == 1
        mock_store.aget_messages.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_messages_cache_hit(self, redis_cache, mock_store):
        """Test aget_messages with cache hit."""
        cached_data = json.dumps([{"role": "user", "content": "Hello"}])
        redis_cache.redis_client.get = MagicMock(return_value=cached_data)
        if hasattr(mock_store, "aget_messages"):
            mock_store.aget_messages = AsyncMock()

        result = await redis_cache.aget_messages(session_id="session1", user_id="user1")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_asave_message(self, redis_cache, mock_store):
        """Test asave_message async method."""
        mock_store.asave_message = AsyncMock()
        mock_store.get_messages = MagicMock(return_value=[{"role": "user", "content": "Hello"}])
        await redis_cache.asave_message("session1", "user", "Hello", user_id="user1")

        mock_store.asave_message.assert_called_once()
        # After saving, it updates the cache with new messages
        redis_cache.redis_client.setex.assert_called_once()

    def test_setup(self, redis_cache, mock_store):
        """Test setup method."""
        redis_cache.setup()
        mock_store.setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_asetup(self, redis_cache, mock_store):
        """Test asetup async method."""
        if hasattr(mock_store, "asetup"):
            mock_store.asetup = AsyncMock()
        await redis_cache.asetup()
        if hasattr(mock_store, "asetup"):
            mock_store.asetup.assert_called_once()
