from unittest.mock import MagicMock, Mock, patch

import pytest

from hypertic.memory.base import BaseMemory
from hypertic.memory.postgres.postgres import PostgresServer
from hypertic.memory.postgres.postgres_async import AsyncPostgresServer


class TestBaseMemory:
    def test_base_memory_is_abstract(self):
        with pytest.raises(TypeError):
            BaseMemory()  # type: ignore[abstract]

    def test_base_memory_has_required_methods(self):
        assert hasattr(BaseMemory, "get_messages")
        assert hasattr(BaseMemory, "save_message")
        assert hasattr(BaseMemory, "setup")
        assert hasattr(BaseMemory, "aget_messages")
        assert hasattr(BaseMemory, "asave_message")
        assert hasattr(BaseMemory, "asetup")


class TestPostgresServer:
    @pytest.fixture
    def postgres_memory(self):
        with patch("hypertic.memory.postgres.postgres.create_engine", return_value=MagicMock()):
            return PostgresServer(db_url="postgresql://localhost/test")

    def test_postgres_server_creation(self, postgres_memory):
        assert postgres_memory is not None
        assert postgres_memory.db_url == "postgresql://localhost/test"

    @pytest.mark.asyncio
    @patch("hypertic.memory.postgres.postgres.create_engine")
    async def test_postgres_server_setup(self, mock_create_engine, postgres_memory):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        postgres_memory.setup()
        assert True


class TestAsyncPostgresServer:
    @pytest.fixture
    def async_postgres_memory(self):
        with patch("hypertic.memory.postgres.postgres_async.create_async_engine", return_value=MagicMock()):
            return AsyncPostgresServer(db_url="postgresql://localhost/test")

    def test_async_postgres_server_creation(self, async_postgres_memory):
        assert async_postgres_memory is not None

    @pytest.mark.asyncio
    @patch("hypertic.memory.postgres.postgres_async.create_async_engine")
    async def test_async_postgres_server_asetup(self, mock_create_engine, async_postgres_memory):
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        await async_postgres_memory.asetup()
        assert True


try:
    from hypertic.memory.redis.async_cache import AsyncRedisCache
    from hypertic.memory.redis.cache import RedisCache

    class TestRedisCache:
        @pytest.fixture
        def redis_cache(self):
            from hypertic.memory.base import BaseMemory

            store = Mock(spec=BaseMemory)
            return RedisCache(store=store, redis_url="redis://localhost:6379")

        def test_redis_cache_creation(self, redis_cache):
            assert redis_cache is not None

    class TestAsyncRedisCache:
        @pytest.fixture
        def async_redis_cache(self):
            from hypertic.memory.base import BaseMemory

            store = Mock(spec=BaseMemory)
            return AsyncRedisCache(store=store, redis_url="redis://localhost:6379")

        def test_async_redis_cache_creation(self, async_redis_cache):
            assert async_redis_cache is not None

except ImportError:
    pass

try:
    from hypertic.memory.mongodb.mongo import MongoServer
    from hypertic.memory.mongodb.mongo_async import AsyncMongoServer

    class TestMongoServer:
        @pytest.fixture
        def mongo_server(self):
            return MongoServer(connection_string="mongodb://localhost:27017")

        def test_mongo_server_creation(self, mongo_server):
            assert mongo_server is not None

    class TestAsyncMongoServer:
        @pytest.fixture
        def async_mongo_server(self):
            return AsyncMongoServer(connection_string="mongodb://localhost:27017")

        def test_async_mongo_server_creation(self, async_mongo_server):
            assert async_mongo_server is not None

except ImportError:
    pass
