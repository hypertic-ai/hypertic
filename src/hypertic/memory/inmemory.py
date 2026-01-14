import threading
from datetime import datetime, timezone
from typing import Any

from hypertic.memory.base import BaseMemory
from hypertic.utils.log import get_logger

logger = get_logger(__name__)


class InMemory(BaseMemory):
    def __init__(self):
        self._data: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._initialized = False

    def setup(self) -> None:
        if not self._initialized:
            logger.debug("InMemory backend initialized")
            self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.setup()

    def get_messages(self, session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        self._ensure_initialized()

        with self._lock:
            filtered_messages = []

            for entry in self._data:
                message_data = entry.get("message", {})
                if not isinstance(message_data, dict):
                    continue

                if message_data.get("type") == "user_data":
                    continue

                if "role" not in message_data or "content" not in message_data:
                    continue

                entry_session_id = entry.get("session_id")
                entry_user_id = entry.get("user_id")

                if session_id:
                    if entry_session_id != session_id:
                        continue
                    if user_id and entry_user_id != user_id:
                        continue
                elif user_id:
                    if entry_user_id != user_id:
                        continue
                else:
                    continue

                message = {
                    "role": message_data.get("role"),
                    "content": message_data.get("content"),
                    "session_id": entry_session_id,
                    "created_at": entry.get("created_at"),
                }

                filtered_messages.append(message)

            filtered_messages.sort(key=lambda x: x.get("created_at") or datetime.min.replace(tzinfo=timezone.utc))

            return filtered_messages

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
        self._ensure_initialized()

        message_data = {"role": role, "content": content}

        entry = {
            "session_id": session_id,
            "user_id": user_id,
            "message": message_data,
            "tool_calls": tool_calls,
            "tool_outputs": tool_outputs,
            "metadata": metadata,
            "created_at": datetime.now(timezone.utc),
        }

        with self._lock:
            self._data.append(entry)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            logger.debug("InMemory data cleared")

    def clear_session(self, session_id: str, user_id: str | None = None) -> None:
        with self._lock:
            self._data = [
                entry for entry in self._data if not (entry.get("session_id") == session_id and (user_id is None or entry.get("user_id") == user_id))
            ]
            logger.debug(f"Cleared session {session_id} (user_id={user_id})")

    def clear_user(self, user_id: str) -> None:
        with self._lock:
            self._data = [entry for entry in self._data if entry.get("user_id") != user_id]
            logger.debug(f"Cleared all messages for user {user_id}")
