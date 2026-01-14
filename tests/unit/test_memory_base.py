import pytest

from hypertic.memory.base import BaseMemory


class TestBaseMemory:
    def test_base_memory_is_abstract(self):
        """Test that BaseMemory is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseMemory()  # type: ignore[abstract]

    def test_base_memory_has_required_methods(self):
        """Test that BaseMemory has required abstract methods."""
        assert hasattr(BaseMemory, "get_messages")
        assert hasattr(BaseMemory, "save_message")
        assert hasattr(BaseMemory, "aget_messages")
        assert hasattr(BaseMemory, "asave_message")
        assert hasattr(BaseMemory, "setup")
        assert hasattr(BaseMemory, "asetup")

    @pytest.mark.asyncio
    async def test_base_memory_aget_messages(self):
        """Test aget_messages async wrapper."""

        class ConcreteMemory(BaseMemory):
            def get_messages(self, session_id: str | None = None, user_id: str | None = None):
                return [{"role": "user", "content": "test"}]

            def save_message(
                self,
                session_id: str,
                role: str,
                content: str,
                user_id: str | None = None,
                tool_calls: list[dict] | None = None,
                tool_outputs: dict | None = None,
                metadata: dict | None = None,
            ):
                pass

            def setup(self):
                pass

        memory = ConcreteMemory()
        messages = await memory.aget_messages(session_id="test_session", user_id="test_user")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_base_memory_asave_message(self):
        """Test asave_message async wrapper."""

        call_args = []

        class ConcreteMemory(BaseMemory):
            def get_messages(self, session_id: str | None = None, user_id: str | None = None):
                return []

            def save_message(
                self,
                session_id: str,
                role: str,
                content: str,
                user_id: str | None = None,
                tool_calls: list[dict] | None = None,
                tool_outputs: dict | None = None,
                metadata: dict | None = None,
            ):
                call_args.append((session_id, role, content, user_id, tool_calls, tool_outputs, metadata))

            def setup(self):
                pass

        memory = ConcreteMemory()
        await memory.asave_message(
            session_id="test_session",
            role="user",
            content="test message",
            user_id="test_user",
            tool_calls=[{"id": "call_1"}],
            tool_outputs={"call_1": "result"},
            metadata={"key": "value"},
        )

        assert len(call_args) == 1
        assert call_args[0][0] == "test_session"
        assert call_args[0][1] == "user"
        assert call_args[0][2] == "test message"
        assert call_args[0][3] == "test_user"
        assert call_args[0][4] == [{"id": "call_1"}]
        assert call_args[0][5] == {"call_1": "result"}
        assert call_args[0][6] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_base_memory_asetup(self):
        """Test asetup async wrapper."""
        setup_called = []

        class ConcreteMemory(BaseMemory):
            def get_messages(self, session_id: str | None = None, user_id: str | None = None):
                return []

            def save_message(
                self,
                session_id: str,
                role: str,
                content: str,
                user_id: str | None = None,
                tool_calls: list[dict] | None = None,
                tool_outputs: dict | None = None,
                metadata: dict | None = None,
            ):
                pass

            def setup(self):
                setup_called.append(True)

        memory = ConcreteMemory()
        await memory.asetup()
        assert len(setup_called) == 1

    def test_base_memory_get_messages_with_none(self):
        """Test get_messages with None parameters."""

        class ConcreteMemory(BaseMemory):
            def get_messages(self, session_id: str | None = None, user_id: str | None = None):
                return []

            def save_message(
                self,
                session_id: str,
                role: str,
                content: str,
                user_id: str | None = None,
                tool_calls: list[dict] | None = None,
                tool_outputs: dict | None = None,
                metadata: dict | None = None,
            ):
                pass

            def setup(self):
                pass

        memory = ConcreteMemory()
        messages = memory.get_messages(session_id=None, user_id=None)
        assert messages == []

    @pytest.mark.asyncio
    async def test_base_memory_aget_messages_with_none(self):
        """Test aget_messages with None parameters."""

        class ConcreteMemory(BaseMemory):
            def get_messages(self, session_id: str | None = None, user_id: str | None = None):
                return [{"role": "system", "content": "default"}]

            def save_message(
                self,
                session_id: str,
                role: str,
                content: str,
                user_id: str | None = None,
                tool_calls: list[dict] | None = None,
                tool_outputs: dict | None = None,
                metadata: dict | None = None,
            ):
                pass

            def setup(self):
                pass

        memory = ConcreteMemory()
        messages = await memory.aget_messages(session_id=None, user_id=None)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
