import pytest

from hypertic.memory.inmemory import InMemory


class TestInMemory:
    @pytest.fixture
    def memory(self):
        return InMemory()

    def test_inmemory_creation(self, memory):
        """Test InMemory initialization."""
        assert memory is not None
        assert memory._data == []
        assert memory._initialized is False

    def test_setup(self, memory):
        """Test setup method."""
        memory.setup()
        assert memory._initialized is True

    def test_get_messages_empty(self, memory):
        """Test get_messages with no messages."""
        result = memory.get_messages(session_id="session1", user_id="user1")
        assert result == []

    def test_get_messages_with_session_id(self, memory):
        """Test get_messages with session_id."""
        memory.setup()
        memory.save_message("session1", "user", "Hello", user_id="user1")
        memory.save_message("session2", "user", "Hi", user_id="user1")

        result = memory.get_messages(session_id="session1", user_id="user1")
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_get_messages_with_user_id_only(self, memory):
        """Test get_messages with user_id only."""
        memory.setup()
        memory.save_message("session1", "user", "Hello", user_id="user1")
        memory.save_message("session2", "user", "Hi", user_id="user1")
        memory.save_message("session3", "user", "Hey", user_id="user2")

        result = memory.get_messages(user_id="user1")
        assert len(result) == 2

    def test_get_messages_filters_invalid_entries(self, memory):
        """Test get_messages filters invalid entries."""
        memory.setup()
        # Add invalid entry directly
        memory._data.append({"session_id": "session1", "user_id": "user1", "message": "invalid"})
        memory.save_message("session1", "user", "Hello", user_id="user1")

        result = memory.get_messages(session_id="session1", user_id="user1")
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_get_messages_filters_user_data_type(self, memory):
        """Test get_messages filters user_data type."""
        memory.setup()
        memory._data.append(
            {
                "session_id": "session1",
                "user_id": "user1",
                "message": {"type": "user_data", "role": "user", "content": "data"},
            }
        )
        memory.save_message("session1", "user", "Hello", user_id="user1")

        result = memory.get_messages(session_id="session1", user_id="user1")
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_save_message(self, memory):
        """Test save_message."""
        memory.setup()
        memory.save_message("session1", "user", "Hello", user_id="user1")

        result = memory.get_messages(session_id="session1", user_id="user1")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_save_message_with_tool_calls(self, memory):
        """Test save_message with tool_calls."""
        memory.setup()
        tool_calls = [{"id": "call1", "type": "function", "function": {"name": "test"}}]
        memory.save_message("session1", "assistant", "Response", user_id="user1", tool_calls=tool_calls)

        result = memory.get_messages(session_id="session1", user_id="user1")
        assert len(result) == 1
        # Check that entry has tool_calls
        entry = memory._data[0]
        assert entry["tool_calls"] == tool_calls

    def test_save_message_with_tool_outputs(self, memory):
        """Test save_message with tool_outputs."""
        memory.setup()
        tool_outputs = {"test": "result"}
        memory.save_message("session1", "tool", "Result", user_id="user1", tool_outputs=tool_outputs)

        entry = memory._data[0]
        assert entry["tool_outputs"] == tool_outputs

    def test_save_message_with_metadata(self, memory):
        """Test save_message with metadata."""
        memory.setup()
        metadata = {"key": "value"}
        memory.save_message("session1", "user", "Hello", user_id="user1", metadata=metadata)

        entry = memory._data[0]
        assert entry["metadata"] == metadata

    def test_clear(self, memory):
        """Test clear method."""
        memory.setup()
        memory.save_message("session1", "user", "Hello", user_id="user1")
        memory.save_message("session2", "user", "Hi", user_id="user2")

        memory.clear()
        assert len(memory._data) == 0

    def test_clear_session(self, memory):
        """Test clear_session method."""
        memory.setup()
        memory.save_message("session1", "user", "Hello", user_id="user1")
        memory.save_message("session2", "user", "Hi", user_id="user1")
        memory.save_message("session1", "assistant", "Response", user_id="user1")

        memory.clear_session("session1", user_id="user1")
        result = memory.get_messages(user_id="user1")
        assert len(result) == 1
        assert result[0]["content"] == "Hi"

    def test_clear_user(self, memory):
        """Test clear_user method."""
        memory.setup()
        memory.save_message("session1", "user", "Hello", user_id="user1")
        memory.save_message("session2", "user", "Hi", user_id="user1")
        memory.save_message("session3", "user", "Hey", user_id="user2")

        memory.clear_user("user1")
        result = memory.get_messages(user_id="user1")
        assert len(result) == 0
        result2 = memory.get_messages(user_id="user2")
        assert len(result2) == 1

    @pytest.mark.asyncio
    async def test_aget_messages(self, memory):
        """Test aget_messages async method."""
        memory.setup()
        memory.save_message("session1", "user", "Hello", user_id="user1")

        result = await memory.aget_messages(session_id="session1", user_id="user1")
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_asave_message(self, memory):
        """Test asave_message async method."""
        memory.setup()
        await memory.asave_message("session1", "user", "Hello", user_id="user1")

        result = await memory.aget_messages(session_id="session1", user_id="user1")
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_asetup(self, memory):
        """Test asetup async method."""
        await memory.asetup()
        assert memory._initialized is True
