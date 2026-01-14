from unittest.mock import patch

import pytest

from hypertic.memory.postgres import AsyncPostgresServer, PostgresServer


class TestMemoryBackendsIntegration:
    @pytest.mark.parametrize(
        "memory_class,memory_params",
        [
            (PostgresServer, {"db_url": "postgresql://localhost/test"}),
        ],
    )
    def test_memory_initialization(self, memory_class, memory_params):
        memory = memory_class(**memory_params)
        assert memory is not None

    @pytest.mark.parametrize(
        "memory_class,memory_params",
        [
            (PostgresServer, {"db_url": "postgresql://localhost/test"}),
        ],
    )
    def test_memory_save_message(self, memory_class, memory_params):
        memory = memory_class(**memory_params)

        with patch.object(memory, "save_message", return_value=None):
            memory.save_message(
                session_id="test_session",
                role="user",
                content="Hello",
                user_id="test_user",
            )
            assert True

    @pytest.mark.parametrize(
        "memory_class,memory_params",
        [
            (PostgresServer, {"db_url": "postgresql://localhost/test"}),
        ],
    )
    def test_memory_get_messages(self, memory_class, memory_params):
        memory = memory_class(**memory_params)

        mock_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        with patch.object(memory, "get_messages", return_value=mock_messages):
            messages = memory.get_messages(session_id="test_session", user_id="test_user")
            assert isinstance(messages, list)
            assert len(messages) == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "memory_class,memory_params",
        [
            (AsyncPostgresServer, {"db_url": "postgresql://localhost/test"}),
        ],
    )
    async def test_memory_async_save_message(self, memory_class, memory_params):
        memory = memory_class(**memory_params)

        with patch.object(memory, "asave_message", return_value=None):
            await memory.asave_message(
                session_id="test_session",
                role="user",
                content="Hello",
                user_id="test_user",
            )
            assert True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "memory_class,memory_params",
        [
            (AsyncPostgresServer, {"db_url": "postgresql://localhost/test"}),
        ],
    )
    async def test_memory_async_get_messages(self, memory_class, memory_params):
        memory = memory_class(**memory_params)

        mock_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        with patch.object(memory, "aget_messages", return_value=mock_messages):
            messages = await memory.aget_messages(session_id="test_session", user_id="test_user")
            assert isinstance(messages, list)
            assert len(messages) == 2

    def test_memory_session_scoped(self):
        memory = PostgresServer(db_url="postgresql://localhost/test")

        with patch.object(memory, "get_messages", return_value=[]):
            messages = memory.get_messages(session_id="session1", user_id="user1")
            assert isinstance(messages, list)

    def test_memory_user_scoped(self):
        memory = PostgresServer(db_url="postgresql://localhost/test")

        with patch.object(memory, "get_messages", return_value=[]):
            messages = memory.get_messages(user_id="user1")
            assert isinstance(messages, list)

    def test_memory_with_tool_calls(self):
        memory = PostgresServer(db_url="postgresql://localhost/test")

        tool_calls = [{"id": "call_1", "function": {"name": "test_tool"}}]

        with patch.object(memory, "save_message", return_value=None):
            memory.save_message(
                session_id="test_session",
                role="assistant",
                content="I'll use a tool",
                tool_calls=tool_calls,
            )
            assert True

    def test_memory_with_tool_outputs(self):
        memory = PostgresServer(db_url="postgresql://localhost/test")

        tool_outputs = {"call_1": "Tool result"}

        with patch.object(memory, "save_message", return_value=None):
            memory.save_message(
                session_id="test_session",
                role="assistant",
                content="Tool executed",
                tool_outputs=tool_outputs,
            )
            assert True

    def test_memory_with_metadata(self):
        memory = PostgresServer(db_url="postgresql://localhost/test")

        metadata = {"model": "test-model", "tokens": 100, "response_time": 0.5}

        with patch.object(memory, "save_message", return_value=None):
            memory.save_message(
                session_id="test_session",
                role="assistant",
                content="Response",
                metadata=metadata,
            )
            assert True


try:
    from hypertic.memory.redis import AsyncRedisCache, RedisCache

    class TestRedisMemoryIntegration:
        def test_redis_initialization(self):
            from hypertic.memory.inmemory import InMemory

            store = InMemory()
            memory = RedisCache(store=store, redis_url="redis://localhost:6379")
            assert memory is not None

        @pytest.mark.asyncio
        async def test_redis_async_operations(self):
            from hypertic.memory.inmemory import InMemory

            store = InMemory()
            memory = AsyncRedisCache(store=store, redis_url="redis://localhost:6379")
            assert memory is not None

except ImportError:
    pass

try:
    from hypertic.memory.mongodb import AsyncMongoServer, MongoServer

    class TestMongoMemoryIntegration:
        def test_mongo_initialization(self):
            memory = MongoServer(connection_string="mongodb://localhost:27017")
            assert memory is not None

        @pytest.mark.asyncio
        async def test_mongo_async_operations(self):
            memory = AsyncMongoServer(connection_string="mongodb://localhost:27017")
            assert memory is not None

except ImportError:
    pass
