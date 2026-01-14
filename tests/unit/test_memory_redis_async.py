from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.memory.inmemory import InMemory
from hypertic.memory.redis.async_cache import AsyncRedisCache


@pytest.mark.unit
class TestAsyncRedisCache:
    @pytest.fixture
    def mock_store(self):
        return InMemory()

    @pytest.fixture
    def async_redis_cache(self, mock_store):
        return AsyncRedisCache(store=mock_store, redis_url="redis://localhost:6379")

    def test_async_redis_cache_creation(self, async_redis_cache, mock_store):
        assert async_redis_cache.store == mock_store
        assert async_redis_cache.redis_url == "redis://localhost:6379"
        assert async_redis_cache.ttl == 3600
        assert async_redis_cache.key_prefix == "cache:memory:"

    @patch("hypertic.memory.redis.async_cache.getenv")
    def test_async_redis_cache_with_env_url(self, mock_getenv):
        mock_getenv.return_value = "redis://env:6379"
        cache = AsyncRedisCache(store=InMemory())
        assert cache.redis_url == "redis://env:6379"

    def test_async_redis_cache_default_url(self):
        """Test default Redis URL."""
        cache = AsyncRedisCache(store=InMemory())
        assert cache.redis_url == "redis://localhost:6379"

    def test_make_cache_key_with_session_id(self, async_redis_cache):
        """Test _make_cache_key with session_id."""
        key = async_redis_cache._make_cache_key(session_id="session1")
        assert key is not None
        assert "cache:memory:" in key
        assert "session1" not in key  # Should be hashed

    def test_make_cache_key_with_user_id(self, async_redis_cache):
        """Test _make_cache_key with user_id."""
        key = async_redis_cache._make_cache_key(user_id="user1")
        assert key is not None
        assert "cache:memory:" in key

    def test_make_cache_key_with_both(self, async_redis_cache):
        """Test _make_cache_key with both session_id and user_id."""
        key1 = async_redis_cache._make_cache_key(session_id="session1", user_id="user1")
        key2 = async_redis_cache._make_cache_key(session_id="session1", user_id="user1")
        assert key1 == key2  # Should be deterministic

    def test_make_cache_key_none(self, async_redis_cache):
        """Test _make_cache_key with no session_id or user_id."""
        key = async_redis_cache._make_cache_key()
        assert key is None

    @pytest.mark.asyncio
    async def test_ensure_redis_connected(self, async_redis_cache):
        """Test _ensure_redis_connected."""
        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(return_value=True)
        with patch("hypertic.memory.redis.async_cache.from_url", new_callable=AsyncMock, return_value=mock_redis):
            await async_redis_cache._ensure_redis_connected()
            assert async_redis_cache.redis_client == mock_redis
            assert async_redis_cache._initialized is True

    @pytest.mark.asyncio
    async def test_ensure_redis_connected_failure(self, async_redis_cache):
        """Test _ensure_redis_connected handles connection failure."""
        from redis.exceptions import RedisError

        with patch("hypertic.memory.redis.async_cache.from_url", side_effect=RedisError("Failed")):
            await async_redis_cache._ensure_redis_connected()
            assert async_redis_cache.redis_client is None
            assert async_redis_cache._initialized is True

    @pytest.mark.asyncio
    async def test_close_redis(self, async_redis_cache):
        """Test _close_redis."""
        mock_redis = MagicMock()
        mock_redis.aclose = AsyncMock()
        async_redis_cache.redis_client = mock_redis
        await async_redis_cache._close_redis()
        mock_redis.aclose.assert_called_once()
        assert async_redis_cache.redis_client is None

    @pytest.mark.asyncio
    async def test_aget_messages_cache_hit(self, async_redis_cache):
        """Test aget_messages with cache hit."""
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value='[{"role": "user", "content": "cached"}]')
        async_redis_cache.redis_client = mock_redis
        async_redis_cache._initialized = True

        messages = await async_redis_cache.aget_messages(session_id="session1")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_aget_messages_cache_miss(self, async_redis_cache):
        """Test aget_messages with cache miss."""
        mock_store = MagicMock()
        mock_store.aget_messages = AsyncMock(return_value=[{"role": "user", "content": "test", "session_id": "s1", "created_at": "2024-01-01"}])
        async_redis_cache.store = mock_store
        mock_redis = MagicMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()
        async_redis_cache.redis_client = mock_redis
        async_redis_cache._initialized = True

        messages = await async_redis_cache.aget_messages(session_id="session1")
        assert len(messages) == 1
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_messages_no_redis_client(self, async_redis_cache):
        """Test aget_messages when Redis client is None."""
        mock_store = MagicMock()
        mock_store.aget_messages = AsyncMock(return_value=[{"role": "user", "content": "test"}])
        async_redis_cache.store = mock_store
        async_redis_cache.redis_client = None
        async_redis_cache._initialized = True

        messages = await async_redis_cache.aget_messages(session_id="session1")
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_asave_message(self, async_redis_cache):
        """Test asave_message."""
        mock_store = MagicMock()
        mock_store.asave_message = AsyncMock()
        mock_store.aget_messages = AsyncMock(return_value=[{"role": "user", "content": "test"}])
        async_redis_cache.store = mock_store
        mock_redis = MagicMock()
        mock_redis.setex = AsyncMock()
        async_redis_cache.redis_client = mock_redis
        async_redis_cache._initialized = True

        await async_redis_cache.asave_message(session_id="session1", role="user", content="test")
        mock_store.asave_message.assert_called_once()
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_asave_message_no_redis_client(self, async_redis_cache):
        """Test asave_message when Redis client is None."""
        mock_store = MagicMock()
        mock_store.asave_message = AsyncMock()
        async_redis_cache.store = mock_store
        async_redis_cache.redis_client = None
        async_redis_cache._initialized = True

        await async_redis_cache.asave_message(session_id="session1", role="user", content="test")
        mock_store.asave_message.assert_called_once()

    def test_get_messages(self, async_redis_cache):
        """Test get_messages delegates to store."""
        mock_store = MagicMock()
        mock_store.get_messages = Mock(return_value=[{"role": "user", "content": "test"}])
        async_redis_cache.store = mock_store
        messages = async_redis_cache.get_messages(session_id="session1")
        assert len(messages) == 1
        mock_store.get_messages.assert_called_once_with(session_id="session1", user_id=None)

    def test_save_message(self, async_redis_cache):
        """Test save_message delegates to store."""
        mock_store = MagicMock()
        mock_store.save_message = Mock()
        async_redis_cache.store = mock_store
        async_redis_cache.save_message(session_id="session1", role="user", content="test")
        mock_store.save_message.assert_called_once()

    def test_setup(self, async_redis_cache):
        """Test setup delegates to store."""
        mock_store = MagicMock()
        mock_store.setup = Mock()
        async_redis_cache.store = mock_store
        async_redis_cache.setup()
        mock_store.setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_asetup(self, async_redis_cache):
        """Test asetup delegates to store."""
        mock_store = MagicMock()
        mock_store.asetup = AsyncMock()
        async_redis_cache.store = mock_store
        await async_redis_cache.asetup()
        mock_store.asetup.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, async_redis_cache):
        """Test async context manager."""
        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()
        with patch("hypertic.memory.redis.async_cache.from_url", new_callable=AsyncMock, return_value=mock_redis):
            async with async_redis_cache:
                assert async_redis_cache._initialized is True
            # _close_redis should be called
            assert async_redis_cache.redis_client is None
