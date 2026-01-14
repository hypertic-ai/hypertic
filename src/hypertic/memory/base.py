import asyncio
from abc import ABC, abstractmethod
from typing import Any


class BaseMemory(ABC):
    @abstractmethod
    def get_messages(self, session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_outputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    async def aget_messages(self, session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_messages, session_id, user_id)

    async def asave_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_outputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.save_message,
            session_id,
            role,
            content,
            user_id,
            tool_calls,
            tool_outputs,
            metadata,
        )

    @abstractmethod
    def setup(self) -> None:
        pass

    async def asetup(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.setup)
