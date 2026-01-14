from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.memory.mongodb.mongo_async import AsyncMongoServer


@pytest.mark.unit
class TestAsyncMongoServer:
    @pytest.fixture
    def async_mongo(self):
        return AsyncMongoServer(connection_string="mongodb://localhost:27017/", database_name="test_db")

    def test_async_mongo_creation(self, async_mongo):
        assert async_mongo.connection_string == "mongodb://localhost:27017/"
        assert async_mongo.database_name == "test_db"
        assert async_mongo.collection_name == "agent_memory"

    @patch("hypertic.memory.mongodb.mongo_async.getenv")
    def test_async_mongo_with_env_connection_string(self, mock_getenv):
        mock_getenv.return_value = "mongodb://env:27017/"
        mongo = AsyncMongoServer()
        assert mongo.connection_string == "mongodb://env:27017/"

    def test_async_mongo_extract_database_from_connection_string(self):
        """Test database name extraction from connection string."""
        mongo = AsyncMongoServer(connection_string="mongodb://localhost:27017/mydb")
        assert mongo.database_name == "mydb"

    def test_async_mongo_default_database(self):
        """Test default database name when not in connection string."""
        mongo = AsyncMongoServer(connection_string="mongodb://localhost:27017/")
        assert mongo.database_name == "hypertic"

    def test_async_mongo_connection_string_cleanup(self):
        """Test connection string cleanup."""
        mongo = AsyncMongoServer(connection_string="mongodb://localhost:27017/mydb")
        assert mongo.connection_string == "mongodb://localhost:27017/"

    @pytest.mark.asyncio
    async def test_get_client(self, async_mongo):
        """Test _get_client creates AsyncIOMotorClient."""
        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock()
        with patch("hypertic.memory.mongodb.mongo_async.AsyncIOMotorClient", return_value=mock_client):
            client = await async_mongo._get_client()
            assert client == mock_client
            mock_client.admin.command.assert_called_once_with("ping")

    @pytest.mark.asyncio
    async def test_get_client_connection_failure(self, async_mongo):
        """Test _get_client handles connection failure."""
        from pymongo.errors import ConnectionFailure

        with patch("hypertic.memory.mongodb.mongo_async.AsyncIOMotorClient", side_effect=ConnectionFailure("Failed")):
            with pytest.raises(ConnectionFailure):
                await async_mongo._get_client()

    @pytest.mark.asyncio
    async def test_get_database(self, async_mongo):
        """Test _get_database."""
        mock_client = MagicMock()
        mock_database = MagicMock()
        mock_client.__getitem__ = Mock(return_value=mock_database)
        async_mongo._client = mock_client
        database = await async_mongo._get_database()
        assert database == mock_database
        mock_client.__getitem__.assert_called_once_with("test_db")

    @pytest.mark.asyncio
    async def test_get_collection(self, async_mongo):
        """Test _get_collection."""
        mock_database = MagicMock()
        mock_collection = MagicMock()
        mock_database.__getitem__ = Mock(return_value=mock_collection)
        async_mongo._database = mock_database
        collection = await async_mongo._get_collection()
        assert collection == mock_collection
        mock_database.__getitem__.assert_called_once_with("agent_memory")

    @pytest.mark.asyncio
    async def test_close(self, async_mongo):
        """Test _close method."""
        mock_client = MagicMock()
        mock_client.close = Mock()
        async_mongo._client = mock_client
        await async_mongo._close()
        mock_client.close.assert_called_once()
        assert async_mongo._client is None
        assert async_mongo._database is None
        assert async_mongo._collection is None

    @pytest.mark.asyncio
    async def test_asetup(self, async_mongo):
        """Test asetup creates indexes."""
        mock_collection = MagicMock()

        # Create async iterator for list_indexes
        async def async_index_iter():
            yield {"name": "idx_agent_memory_session_created"}

        mock_collection.list_indexes = Mock(return_value=async_index_iter())
        mock_collection.create_index = AsyncMock()

        async def get_collection():
            return mock_collection

        async_mongo._get_collection = get_collection
        await async_mongo.asetup()
        assert async_mongo._initialized is True

    @pytest.mark.asyncio
    async def test_asetup_indexes_exist(self, async_mongo):
        """Test asetup when indexes already exist."""
        mock_collection = MagicMock()
        mock_indexes = [
            {"name": "idx_agent_memory_session_created"},
            {"name": "idx_agent_memory_user_id"},
            {"name": "idx_agent_memory_session_id"},
        ]

        async def async_index_iter():
            for idx in mock_indexes:
                yield idx

        mock_collection.list_indexes = Mock(return_value=async_index_iter())

        async def get_collection():
            return mock_collection

        async_mongo._get_collection = get_collection
        await async_mongo.asetup()
        assert async_mongo._initialized is True
        mock_collection.create_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_asetup_already_initialized(self, async_mongo):
        """Test asetup when already initialized."""
        async_mongo._initialized = True
        await async_mongo.asetup()
        # Should return early without doing anything

    @pytest.mark.asyncio
    async def test_asetup_index_already_exists_error(self, async_mongo):
        """Test asetup handles index already exists error."""
        mock_collection = MagicMock()

        async def async_index_iter():
            return
            yield  # Make it an async generator

        mock_collection.list_indexes = Mock(return_value=async_index_iter())
        mock_collection.create_index = AsyncMock(side_effect=Exception("already exists"))

        async def get_collection():
            return mock_collection

        async_mongo._get_collection = get_collection
        await async_mongo.asetup()
        assert async_mongo._initialized is True

    @pytest.mark.asyncio
    async def test_aget_messages_by_session_id(self, async_mongo):
        """Test aget_messages with session_id."""
        mock_collection = MagicMock()
        mock_doc = MagicMock()
        mock_doc.get = Mock(
            side_effect=lambda key, default=None: {
                "_id": "doc1",
                "session_id": "session1",
                "created_at": "2024-01-01",
                "message": {"role": "user", "content": "test"},
            }.get(key, default)
        )

        async def async_cursor_iter():
            yield mock_doc

        mock_cursor = MagicMock()
        mock_cursor.sort = Mock(return_value=mock_cursor)
        mock_cursor.__aiter__ = Mock(return_value=async_cursor_iter())
        mock_collection.find = Mock(return_value=mock_cursor)

        async def get_collection():
            return mock_collection

        async_mongo._get_collection = get_collection
        async_mongo._initialized = True
        messages = await async_mongo.aget_messages(session_id="session1")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_aget_messages_by_user_id(self, async_mongo):
        """Test aget_messages with user_id."""
        mock_collection = MagicMock()
        mock_doc = MagicMock()
        mock_doc.get = Mock(
            side_effect=lambda key, default=None: {
                "_id": "doc1",
                "session_id": "session1",
                "created_at": "2024-01-01",
                "message": {"role": "assistant", "content": "response"},
            }.get(key, default)
        )

        async def async_cursor_iter():
            yield mock_doc

        mock_cursor = MagicMock()
        mock_cursor.sort = Mock(return_value=mock_cursor)
        mock_cursor.__aiter__ = Mock(return_value=async_cursor_iter())
        mock_collection.find = Mock(return_value=mock_cursor)

        async def get_collection():
            return mock_collection

        async_mongo._get_collection = get_collection
        async_mongo._initialized = True
        messages = await async_mongo.aget_messages(user_id="user1")
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_aget_messages_no_session_or_user(self, async_mongo):
        """Test aget_messages with no session_id or user_id."""
        async_mongo._initialized = True
        messages = await async_mongo.aget_messages()
        assert messages == []

    @pytest.mark.asyncio
    async def test_aget_messages_skip_user_data(self, async_mongo):
        """Test aget_messages skips user_data type messages."""
        mock_collection = MagicMock()
        mock_doc = MagicMock()
        mock_doc.get = Mock(
            side_effect=lambda key, default=None: {
                "_id": "doc1",
                "session_id": "session1",
                "created_at": "2024-01-01",
                "message": {"type": "user_data", "role": "user", "content": "test"},
            }.get(key, default)
        )
        mock_cursor = MagicMock()
        mock_cursor.sort = Mock(return_value=mock_cursor)
        mock_cursor.__aiter__ = Mock(return_value=iter([mock_doc]))
        mock_collection.find = Mock(return_value=mock_cursor)

        async def get_collection():
            return mock_collection

        async_mongo._get_collection = get_collection
        async_mongo._initialized = True
        messages = await async_mongo.aget_messages(session_id="session1")
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_aget_messages_invalid_message_format(self, async_mongo):
        """Test aget_messages handles invalid message format."""
        mock_collection = MagicMock()
        mock_doc = MagicMock()
        mock_doc.get = Mock(
            side_effect=lambda key, default=None: {
                "_id": "doc1",
                "session_id": "session1",
                "created_at": "2024-01-01",
                "message": "invalid_string",
            }.get(key, default)
        )
        mock_cursor = MagicMock()
        mock_cursor.sort = Mock(return_value=mock_cursor)
        mock_cursor.__aiter__ = Mock(return_value=iter([mock_doc]))
        mock_collection.find = Mock(return_value=mock_cursor)

        async def get_collection():
            return mock_collection

        async_mongo._get_collection = get_collection
        async_mongo._initialized = True
        messages = await async_mongo.aget_messages(session_id="session1")
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_asave_message(self, async_mongo):
        """Test asave_message."""
        mock_collection = MagicMock()
        mock_collection.insert_one = AsyncMock()

        async def get_collection():
            return mock_collection

        async_mongo._get_collection = get_collection
        async_mongo._initialized = True
        await async_mongo.asave_message(session_id="session1", role="user", content="test message", user_id="user1")
        mock_collection.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_asave_message_with_tool_calls(self, async_mongo):
        """Test asave_message with tool_calls."""
        mock_collection = MagicMock()
        mock_collection.insert_one = AsyncMock()

        async def get_collection():
            return mock_collection

        async_mongo._get_collection = get_collection
        async_mongo._initialized = True
        await async_mongo.asave_message(
            session_id="session1",
            role="assistant",
            content="test",
            tool_calls=[{"name": "tool1", "arguments": {}}],
        )
        call_args = mock_collection.insert_one.call_args[0][0]
        assert "tool_calls" in call_args

    def test_get_messages(self, async_mongo):
        """Test get_messages sync wrapper."""
        with patch.object(async_mongo, "aget_messages", new_callable=AsyncMock, return_value=[{"role": "user", "content": "test"}]):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_until_complete = Mock(return_value=[{"role": "user", "content": "test"}])
                messages = async_mongo.get_messages(session_id="session1")
                assert len(messages) == 1

    def test_save_message(self, async_mongo):
        """Test save_message sync wrapper."""
        with patch.object(async_mongo, "asave_message", new_callable=AsyncMock):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_until_complete = Mock()
                async_mongo.save_message(session_id="session1", role="user", content="test")
                mock_loop.return_value.run_until_complete.assert_called_once()

    def test_setup(self, async_mongo):
        """Test setup sync wrapper."""
        with patch.object(async_mongo, "asetup", new_callable=AsyncMock):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_until_complete = Mock()
                async_mongo.setup()
                mock_loop.return_value.run_until_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_context_manager(self):
        """Test create class method as context manager."""
        with patch("hypertic.memory.mongodb.mongo_async.AsyncIOMotorClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.admin.command = AsyncMock()
            mock_client.close = Mock()
            mock_client_class.return_value = mock_client
            async with AsyncMongoServer.create("mongodb://localhost:27017/", database_name="test") as mongo:
                assert mongo.database_name == "test"
            # close() is called in _close() which is called in the finally block
            # Since it's not awaited, we just check that _close was called
            assert mongo._client is None
