from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.memory.postgres.postgres_async import AsyncPostgresServer


@pytest.mark.unit
class TestAsyncPostgresServer:
    @pytest.fixture
    def async_postgres(self):
        return AsyncPostgresServer(db_url="postgresql+asyncpg://localhost/test", table_name="test_memory")

    def test_async_postgres_creation(self, async_postgres):
        assert async_postgres.db_url == "postgresql+asyncpg://localhost/test"
        assert async_postgres.table_name == "test_memory"

    @patch("hypertic.memory.postgres.postgres_async.getenv")
    def test_async_postgres_with_env_db_url(self, mock_getenv):
        mock_getenv.return_value = "postgresql://localhost/test"
        postgres = AsyncPostgresServer()
        assert "postgresql+asyncpg://" in postgres.db_url

    def test_async_postgres_convert_postgresql_url(self):
        """Test conversion of postgresql:// to postgresql+asyncpg://."""
        postgres = AsyncPostgresServer(db_url="postgresql://localhost/test")
        assert postgres.db_url == "postgresql+asyncpg://localhost/test"

    def test_async_postgres_no_db_url(self):
        """Test error when no db_url provided."""
        with patch("hypertic.memory.postgres.postgres_async.getenv", return_value=None):
            with pytest.raises(ValueError, match="db_url is required"):
                AsyncPostgresServer()

    def test_async_postgres_invalid_db_url(self):
        """Test error with invalid db_url."""
        with pytest.raises(ValueError, match="asyncpg driver"):
            AsyncPostgresServer(db_url="mysql://localhost/test")

    @pytest.mark.asyncio
    async def test_asetup_table_exists(self, async_postgres):
        """Test asetup when table already exists."""
        # Mock the engine.begin() method by patching the engine
        mock_conn = MagicMock()
        mock_conn.run_sync = AsyncMock(return_value=True)

        class AsyncContextManager:
            async def __aenter__(self):
                return mock_conn

            async def __aexit__(self, *args):
                return None

        mock_engine = MagicMock()
        mock_engine.begin = Mock(return_value=AsyncContextManager())
        async_postgres.engine = mock_engine
        await async_postgres.asetup()
        assert async_postgres._initialized is True

    @pytest.mark.asyncio
    async def test_asetup_create_table(self, async_postgres):
        """Test asetup creates table."""
        # Mock the engine.begin() method by patching the engine
        mock_conn = MagicMock()
        mock_conn.run_sync = AsyncMock(side_effect=[False, None, True])

        class AsyncContextManager:
            async def __aenter__(self):
                return mock_conn

            async def __aexit__(self, *args):
                return None

        mock_engine = MagicMock()
        mock_engine.begin = Mock(return_value=AsyncContextManager())
        async_postgres.engine = mock_engine
        with patch("hypertic.memory.postgres.postgres_async.SQLAlchemyBase") as mock_base:
            mock_base.metadata.create_all = Mock()
            await async_postgres.asetup()
            assert async_postgres._initialized is True

    @pytest.mark.asyncio
    async def test_asetup_already_initialized(self, async_postgres):
        """Test asetup when already initialized."""
        async_postgres._initialized = True
        await async_postgres.asetup()
        # Should return early

    @pytest.mark.asyncio
    async def test_aget_messages_by_session_id(self, async_postgres):
        """Test aget_messages with session_id."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.id = 1
        mock_row.session_id = "session1"
        mock_row.created_at = "2024-01-01"
        mock_row.message = {"role": "user", "content": "test"}
        mock_result.scalars.return_value.all.return_value = [mock_row]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        async_postgres.AsyncSession = Mock(return_value=mock_session)
        async_postgres._initialized = True

        messages = await async_postgres.aget_messages(session_id="session1")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_aget_messages_by_user_id(self, async_postgres):
        """Test aget_messages with user_id."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.id = 1
        mock_row.session_id = "session1"
        mock_row.created_at = "2024-01-01"
        mock_row.message = {"role": "assistant", "content": "response"}
        mock_result.scalars.return_value.all.return_value = [mock_row]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        async_postgres.AsyncSession = Mock(return_value=mock_session)
        async_postgres._initialized = True

        messages = await async_postgres.aget_messages(user_id="user1")
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_aget_messages_no_session_or_user(self, async_postgres):
        """Test aget_messages with no session_id or user_id."""
        async_postgres._initialized = True
        messages = await async_postgres.aget_messages()
        assert messages == []

    @pytest.mark.asyncio
    async def test_aget_messages_skip_user_data(self, async_postgres):
        """Test aget_messages skips user_data type messages."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.id = 1
        mock_row.message = {"type": "user_data", "role": "user", "content": "test"}
        mock_result.scalars.return_value.all.return_value = [mock_row]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        async_postgres.AsyncSession = Mock(return_value=mock_session)
        async_postgres._initialized = True

        messages = await async_postgres.aget_messages(session_id="session1")
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_aget_messages_invalid_message_format(self, async_postgres):
        """Test aget_messages handles invalid message format."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row.id = 1
        mock_row.message = "invalid_string"
        mock_result.scalars.return_value.all.return_value = [mock_row]
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        async_postgres.AsyncSession = Mock(return_value=mock_session)
        async_postgres._initialized = True

        messages = await async_postgres.aget_messages(session_id="session1")
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_asave_message(self, async_postgres):
        """Test asave_message."""
        mock_session = MagicMock()
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        async_postgres.AsyncSession = Mock(return_value=mock_session)
        async_postgres._initialized = True

        await async_postgres.asave_message(session_id="session1", role="user", content="test", user_id="user1")
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_asave_message_with_tool_calls(self, async_postgres):
        """Test asave_message with tool_calls."""
        mock_session = MagicMock()
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        async_postgres.AsyncSession = Mock(return_value=mock_session)
        async_postgres._initialized = True

        await async_postgres.asave_message(
            session_id="session1",
            role="assistant",
            content="test",
            tool_calls=[{"name": "tool1", "arguments": {}}],
        )
        call_args = mock_session.add.call_args[0][0]
        assert hasattr(call_args, "tool_calls")

    def test_get_messages_no_running_loop(self, async_postgres):
        """Test get_messages when no running loop."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run", return_value=[{"role": "user", "content": "test"}]):
                messages = async_postgres.get_messages(session_id="session1")
                assert len(messages) == 1

    def test_get_messages_with_running_loop(self, async_postgres):
        """Test get_messages raises error when called from async context."""
        with patch("asyncio.get_running_loop", return_value=MagicMock()):
            with pytest.raises(RuntimeError, match="Cannot call get_messages"):
                async_postgres.get_messages(session_id="session1")

    def test_save_message_no_running_loop(self, async_postgres):
        """Test save_message when no running loop."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run"):
                async_postgres.save_message(session_id="session1", role="user", content="test")

    def test_save_message_with_running_loop(self, async_postgres):
        """Test save_message raises error when called from async context."""
        with patch("asyncio.get_running_loop", return_value=MagicMock()):
            with pytest.raises(RuntimeError, match="Cannot call save_message"):
                async_postgres.save_message(session_id="session1", role="user", content="test")

    def test_setup_no_running_loop(self, async_postgres):
        """Test setup when no running loop."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run"):
                async_postgres.setup()

    def test_setup_with_running_loop(self, async_postgres):
        """Test setup raises error when called from async context."""
        with patch("asyncio.get_running_loop", return_value=MagicMock()):
            with pytest.raises(RuntimeError, match="Cannot call setup"):
                async_postgres.setup()

    @pytest.mark.asyncio
    async def test_create_context_manager(self):
        """Test create class method as context manager."""
        with patch("hypertic.memory.postgres.postgres_async.create_async_engine") as mock_engine:
            mock_engine_instance = MagicMock()
            mock_engine_instance.dispose = AsyncMock()
            mock_engine.return_value = mock_engine_instance
            async with AsyncPostgresServer.create("postgresql+asyncpg://localhost/test") as postgres:
                assert postgres.db_url == "postgresql+asyncpg://localhost/test"
            mock_engine_instance.dispose.assert_called_once()
