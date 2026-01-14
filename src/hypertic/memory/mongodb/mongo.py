"""
MongoDB memory backend for Hypertic.

Uses pymongo for conversation history storage.
"""

import json
from datetime import datetime, timezone
from os import getenv
from typing import Any

from hypertic.memory.base import BaseMemory
from hypertic.utils.log import get_logger, mask_connection_string

logger = get_logger(__name__)

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.database import Database
    from pymongo.errors import ConnectionFailure
except ImportError as err:
    raise ImportError("pymongo required for MongoServer. Install with: pip install 'pymongo>=4.0.0'") from err


class MongoServer(BaseMemory):
    def __init__(
        self,
        connection_string: str | None = None,
        database_name: str | None = None,
        collection_name: str = "agent_memory",
    ):
        connection_string = connection_string or getenv("MONGODB_URL") or "mongodb://localhost:27017/"

        if database_name is None:
            parts = connection_string.rstrip("/").split("/")
            if len(parts) > 3 and parts[-1] and not parts[-1].startswith("?"):
                database_name = parts[-1].split("?")[0]
            else:
                database_name = "hypertic"

        base_connection = connection_string
        if "/" in base_connection and not base_connection.endswith("/"):
            parts = base_connection.rstrip("/").split("/")
            if len(parts) > 3:
                base_connection = "/".join(parts[:-1]) + "/"

        self.connection_string = base_connection
        self.database_name = database_name
        self.collection_name = collection_name
        self._client: MongoClient[Any] | None = None
        self._database: Database[Any] | None = None
        self._collection: Collection[Any] | None = None
        self._initialized = False

    def _get_client(self) -> MongoClient[Any]:
        if self._client is None:
            try:
                self._client = MongoClient(self.connection_string)
                self._client.admin.command("ping")
                logger.info(f"MongoDB connected: {mask_connection_string(self.connection_string)}")
            except ConnectionFailure as e:
                logger.error(f"MongoDB connection failed: {e}")
                raise
        return self._client

    def _get_database(self) -> Database[Any]:
        if self._database is None:
            client = self._get_client()
            self._database = client[self.database_name]
        return self._database

    def _get_collection(self) -> Collection[Any]:
        if self._collection is None:
            db = self._get_database()
            self._collection = db[self.collection_name]
        return self._collection

    def setup(self) -> None:
        if not self._initialized:
            try:
                collection = self._get_collection()

                idx_prefix = f"idx_{self.collection_name}"
                required_indexes = [
                    f"{idx_prefix}_session_created",
                    f"{idx_prefix}_user_id",
                    f"{idx_prefix}_session_id",
                ]
                existing_indexes = [idx["name"] for idx in collection.list_indexes()]
                indexes_exist = all(idx in existing_indexes for idx in required_indexes)

                if indexes_exist:
                    logger.debug(f"Indexes already exist for {self.collection_name}")
                    self._initialized = True
                    return

                collection.create_index([("session_id", 1), ("created_at", 1)], name=f"{idx_prefix}_session_created")

                collection.create_index([("user_id", 1)], name=f"{idx_prefix}_user_id")

                collection.create_index([("session_id", 1)], name=f"{idx_prefix}_session_id")

                logger.info(f"MongoServer collection ({self.collection_name}) indexes created successfully")
                self._initialized = True
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    logger.debug(f"Indexes already exist for {self.collection_name}")
                    self._initialized = True
                else:
                    logger.error(f"Error creating indexes: {e}", exc_info=True)
                    raise

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.setup()

    def get_messages(self, session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        self._ensure_initialized()

        try:
            collection = self._get_collection()

            query = {}
            if session_id:
                query["session_id"] = session_id
            elif user_id:
                query["user_id"] = user_id
            else:
                return []

            cursor = collection.find(query).sort("created_at", 1)

            messages = []
            for doc in cursor:
                message_data = doc.get("message", {})
                if not isinstance(message_data, dict):
                    try:
                        message_data = json.loads(str(message_data)) if message_data else {}
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Invalid message format for id {doc.get('_id')}, skipping")
                        continue

                if message_data.get("type") == "user_data":
                    continue

                if "role" not in message_data or "content" not in message_data:
                    logger.warning(f"Invalid message format for id {doc.get('_id')}, skipping")
                    continue

                message = {
                    "role": message_data.get("role"),
                    "content": message_data.get("content"),
                    "session_id": doc.get("session_id"),
                    "created_at": doc.get("created_at"),
                }

                messages.append(message)

            return messages
        except Exception as e:
            logger.error(f"Error getting messages: {e}", exc_info=True)
            return []

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

        document = {
            "session_id": session_id,
            "user_id": user_id,
            "message": message_data,
            "tool_calls": tool_calls,
            "tool_outputs": tool_outputs,
            "response_metadata": metadata,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        try:
            collection = self._get_collection()
            collection.insert_one(document)
        except Exception as e:
            logger.error(f"Error saving message for session {session_id}: {e}", exc_info=True)
            raise
