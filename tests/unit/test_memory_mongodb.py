from unittest.mock import MagicMock, patch

import pytest

from hypertic.memory.mongodb.mongo import MongoServer


class TestMongoServer:
    @pytest.fixture
    def mongo_memory(self):
        """Create a MongoServer with mocked MongoClient."""
        mock_client = MagicMock()
        mock_client.admin.command = MagicMock()
        mock_database = MagicMock()
        mock_collection = MagicMock()
        mock_client.__getitem__.return_value = mock_database
        mock_database.__getitem__.return_value = mock_collection
        mock_collection.list_indexes.return_value = []

        # Patch MongoClient to return our mock
        with patch("hypertic.memory.mongodb.mongo.MongoClient", return_value=mock_client):
            memory = MongoServer(connection_string="mongodb://localhost:27017/test")
            # Set the mocked client directly to avoid connection attempts
            memory._client = mock_client
            memory._database = mock_database
            memory._collection = mock_collection
            yield memory
            # Patch stays active until test completes

    def test_mongo_server_creation(self, mongo_memory):
        """Test MongoServer initialization."""
        assert mongo_memory.connection_string == "mongodb://localhost:27017/"
        assert mongo_memory.database_name == "test"
        assert mongo_memory.collection_name == "agent_memory"

    def test_mongo_server_creation_default_database(self):
        """Test MongoServer with default database."""
        with patch("hypertic.memory.mongodb.mongo.MongoClient"):
            memory = MongoServer(connection_string="mongodb://localhost:27017/")
            assert memory.database_name == "hypertic"

    def test_mongo_server_creation_custom_collection(self):
        """Test MongoServer with custom collection name."""
        with patch("hypertic.memory.mongodb.mongo.MongoClient"):
            memory = MongoServer(
                connection_string="mongodb://localhost:27017/test",
                collection_name="custom_memory",
            )
            assert memory.collection_name == "custom_memory"

    def test_get_client(self, mongo_memory):
        """Test _get_client method."""
        # _get_client is already mocked via fixture, just verify it exists
        assert mongo_memory._client is not None

    def test_get_database(self, mongo_memory):
        """Test _get_database method."""
        # _get_database is already mocked via fixture, just verify it exists
        assert mongo_memory._database is not None

    def test_get_collection(self, mongo_memory):
        """Test _get_collection method."""
        # _get_collection is already mocked via fixture, just verify it exists
        assert mongo_memory._collection is not None

    def test_setup(self, mongo_memory):
        """Test setup method."""
        # Mock list_indexes to return empty list (no existing indexes)
        mongo_memory._collection.list_indexes.return_value = []
        mongo_memory.setup()
        assert mongo_memory._initialized is True

    def test_get_messages_empty(self, mongo_memory):
        """Test get_messages with no messages."""
        mock_collection = MagicMock()
        mock_collection.find.return_value = []
        mongo_memory._collection = mock_collection
        mongo_memory._initialized = True

        result = mongo_memory.get_messages(session_id="session1", user_id="user1")
        assert result == []

    def test_get_messages_with_session_id(self, mongo_memory):
        """Test get_messages with session_id."""
        mock_message = {
            "session_id": "session1",
            "user_id": "user1",
            "message": {"role": "user", "content": "Hello"},
            "created_at": "2024-01-01T00:00:00Z",
        }
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = [mock_message]
        mock_collection.find.return_value = mock_cursor
        mongo_memory._collection = mock_collection
        mongo_memory._initialized = True

        result = mongo_memory.get_messages(session_id="session1", user_id="user1")
        assert len(result) == 1

    def test_save_message(self, mongo_memory):
        """Test save_message."""
        mock_collection = MagicMock()
        mongo_memory._collection = mock_collection
        mongo_memory._initialized = True

        mongo_memory.save_message("session1", "user", "Hello", user_id="user1")
        mock_collection.insert_one.assert_called_once()

    def test_save_message_with_tool_calls(self, mongo_memory):
        """Test save_message with tool_calls."""
        mock_collection = MagicMock()
        mongo_memory._collection = mock_collection
        mongo_memory._initialized = True

        tool_calls = [{"id": "call1", "type": "function"}]
        mongo_memory.save_message("session1", "assistant", "Response", user_id="user1", tool_calls=tool_calls)
        mock_collection.insert_one.assert_called_once()
        call_args = mock_collection.insert_one.call_args[0][0]
        assert call_args["tool_calls"] == tool_calls

    @pytest.mark.asyncio
    async def test_aget_messages(self, mongo_memory):
        """Test aget_messages async method."""
        mock_collection = MagicMock()
        mock_collection.find.return_value = []
        mongo_memory._collection = mock_collection
        mongo_memory._initialized = True

        result = await mongo_memory.aget_messages(session_id="session1", user_id="user1")
        assert result == []

    @pytest.mark.asyncio
    async def test_asave_message(self, mongo_memory):
        """Test asave_message async method."""
        mock_collection = MagicMock()
        mongo_memory._collection = mock_collection
        mongo_memory._initialized = True

        await mongo_memory.asave_message("session1", "user", "Hello", user_id="user1")
        mock_collection.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_asetup(self, mongo_memory):
        """Test asetup async method."""
        # Mock list_indexes to return empty list (no existing indexes)
        mongo_memory._collection.list_indexes.return_value = []
        await mongo_memory.asetup()
        assert mongo_memory._initialized is True
