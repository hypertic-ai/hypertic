from unittest.mock import MagicMock, patch

import pytest

from hypertic.memory.postgres.postgres import PostgresServer


class TestPostgresServer:
    @pytest.fixture
    def postgres_memory(self):
        with patch("hypertic.memory.postgres.postgres.create_engine") as mock_engine:
            mock_engine_instance = MagicMock()
            mock_engine.return_value = mock_engine_instance
            with patch("hypertic.memory.postgres.postgres.sessionmaker") as mock_sessionmaker:
                mock_session = MagicMock()
                mock_sessionmaker.return_value = mock_session
                return PostgresServer(db_url="postgresql://localhost/test")

    def test_postgres_server_creation(self, postgres_memory):
        """Test PostgresServer initialization."""
        assert postgres_memory.db_url == "postgresql://localhost/test"
        assert postgres_memory.table_name == "agent_memory"

    def test_postgres_server_creation_custom_table(self):
        """Test PostgresServer with custom table name."""
        with patch("hypertic.memory.postgres.postgres.create_engine"):
            with patch("hypertic.memory.postgres.postgres.sessionmaker"):
                memory = PostgresServer(db_url="postgresql://localhost/test", table_name="custom_memory")
                assert memory.table_name == "custom_memory"

    def test_postgres_server_creation_no_db_url(self):
        """Test PostgresServer raises error when db_url is not provided."""
        with patch("hypertic.memory.postgres.postgres.getenv", return_value=None):
            with pytest.raises(ValueError, match="db_url is required"):
                PostgresServer()

    def test_setup(self, postgres_memory):
        """Test setup method."""
        with patch("hypertic.memory.postgres.postgres.inspect") as mock_inspect:
            mock_inspector = MagicMock()
            mock_inspector.has_table = MagicMock(return_value=False)
            mock_inspect.return_value = mock_inspector
            postgres_memory.setup()
            assert postgres_memory._initialized is True

    def test_get_messages_empty(self, postgres_memory):
        """Test get_messages with no messages."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        postgres_memory.Session.return_value.__enter__.return_value = mock_session
        postgres_memory.Session.return_value.__exit__.return_value = None
        mock_session.scalars.return_value = mock_query
        postgres_memory._initialized = True

        result = postgres_memory.get_messages(session_id="session1", user_id="user1")
        assert result == []

    def test_get_messages_with_session_id(self, postgres_memory):
        """Test get_messages with session_id."""
        mock_message = MagicMock()
        mock_message.message = {"role": "user", "content": "Hello"}
        mock_message.session_id = "session1"
        mock_message.created_at = None
        mock_message.id = 1

        mock_session = MagicMock()
        mock_execute_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_message]
        mock_execute_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_execute_result
        postgres_memory.Session.return_value.__enter__.return_value = mock_session
        postgres_memory.Session.return_value.__exit__.return_value = None
        postgres_memory._initialized = True

        result = postgres_memory.get_messages(session_id="session1", user_id="user1")
        assert len(result) == 1

    def test_save_message(self, postgres_memory):
        """Test save_message."""
        mock_session = MagicMock()
        postgres_memory.Session.return_value.__enter__.return_value = mock_session
        postgres_memory.Session.return_value.__exit__.return_value = None
        postgres_memory._initialized = True

        postgres_memory.save_message("session1", "user", "Hello", user_id="user1")
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_save_message_with_tool_calls(self, postgres_memory):
        """Test save_message with tool_calls."""
        mock_session = MagicMock()
        postgres_memory.Session.return_value.__enter__.return_value = mock_session
        postgres_memory.Session.return_value.__exit__.return_value = None
        postgres_memory._initialized = True

        tool_calls = [{"id": "call1", "type": "function"}]
        postgres_memory.save_message("session1", "assistant", "Response", user_id="user1", tool_calls=tool_calls)
        mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_messages(self, postgres_memory):
        """Test aget_messages async method."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        postgres_memory.Session.return_value.__aenter__.return_value = mock_session
        postgres_memory.Session.return_value.__aexit__.return_value = None
        mock_session.scalars.return_value = mock_query
        postgres_memory._initialized = True

        result = await postgres_memory.aget_messages(session_id="session1", user_id="user1")
        assert result == []

    @pytest.mark.asyncio
    async def test_asave_message(self, postgres_memory):
        """Test asave_message async method."""
        # PostgresServer doesn't have asave_message, it uses the base wrapper
        # So we test that it calls save_message
        with patch.object(postgres_memory, "save_message") as mock_save:
            await postgres_memory.asave_message("session1", "user", "Hello", user_id="user1")
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_asetup(self, postgres_memory):
        """Test asetup async method."""
        with patch("hypertic.memory.postgres.postgres.inspect") as mock_inspect:
            mock_inspector = MagicMock()
            mock_inspector.has_table = MagicMock(return_value=False)
            mock_inspect.return_value = mock_inspector
            await postgres_memory.asetup()
            assert postgres_memory._initialized is True
