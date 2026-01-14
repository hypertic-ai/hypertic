import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from os import getenv
from typing import Any

from hypertic.memory.base import BaseMemory
from hypertic.utils.log import get_logger, mask_connection_string

logger = get_logger(__name__)

try:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase
    from pymongo.errors import ConnectionFailure
except ImportError as err:
    raise ImportError("motor required for AsyncMongoServer. Install with: pip install 'motor>=3.0.0'") from err


class AsyncMongoServer(BaseMemory):
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
        self._client: AsyncIOMotorClient[Any] | None = None
        self._database: AsyncIOMotorDatabase[Any] | None = None
        self._collection: AsyncIOMotorCollection[Any] | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        connection_string: str,
        *,
        database_name: str | None = None,
        collection_name: str = "agent_memory",
    ):
        instance = cls(
            connection_string=connection_string,
            database_name=database_name,
            collection_name=collection_name,
        )
        try:
            yield instance
        finally:
            await instance._close()

    async def _get_client(self) -> AsyncIOMotorClient[Any]:
        if self._client is None:
            try:
                self._client = AsyncIOMotorClient(
                    self.connection_string,
                    maxPoolSize=100,
                    retryWrites=True,
                    serverSelectionTimeoutMS=5000,
                )
                await self._client.admin.command("ping")
                logger.info(f"AsyncMongoDB connected: {mask_connection_string(self.connection_string)}")
            except ConnectionFailure as e:
                logger.error(f"MongoDB connection failed: {e}")
                raise
        return self._client

    async def _get_database(self) -> AsyncIOMotorDatabase[Any]:
        if self._database is None:
            client = await self._get_client()
            self._database = client[self.database_name]
        return self._database

    async def _get_collection(self) -> AsyncIOMotorCollection[Any]:
        if self._collection is None:
            db = await self._get_database()
            self._collection = db[self.collection_name]
        return self._collection

    async def _close(self) -> None:
        if self._client:
            try:
                self._client.close()
                logger.debug("AsyncMongoDB connection closed")
            except Exception as e:
                logger.warning(f"Error closing MongoDB connection: {e}")
            finally:
                self._client = None
                self._database = None
                self._collection = None
                self._initialized = False

    async def asetup(self) -> None:
        async with self._lock:
            if self._initialized:
                return

            try:
                collection = await self._get_collection()

                idx_prefix = f"idx_{self.collection_name}"
                required_indexes = [
                    f"{idx_prefix}_session_created",
                    f"{idx_prefix}_user_id",
                    f"{idx_prefix}_session_id",
                ]
                existing_indexes = []
                async for idx in collection.list_indexes():
                    existing_indexes.append(idx["name"])

                indexes_exist = all(idx in existing_indexes for idx in required_indexes)

                if indexes_exist:
                    logger.debug(f"Indexes already exist for {self.collection_name}")
                    self._initialized = True
                    return

                await collection.create_index([("session_id", 1), ("created_at", 1)], name=f"{idx_prefix}_session_created")

                await collection.create_index([("user_id", 1)], name=f"{idx_prefix}_user_id")

                await collection.create_index([("session_id", 1)], name=f"{idx_prefix}_session_id")

                logger.info(f"AsyncMongoServer collection ({self.collection_name}) indexes created successfully")
                self._initialized = True
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    logger.debug(f"Indexes already exist for {self.collection_name}")
                    self._initialized = True
                else:
                    logger.error(f"Error creating indexes: {e}", exc_info=True)
                    raise

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.asetup()

    def get_messages(self, session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aget_messages(session_id=session_id, user_id=user_id))

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
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.asave_message(session_id, role, content, user_id, tool_calls, tool_outputs, metadata))

    async def aget_messages(self, session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        await self._ensure_initialized()

        try:
            collection = await self._get_collection()

            query = {}
            if session_id:
                query["session_id"] = session_id
            elif user_id:
                query["user_id"] = user_id
            else:
                return []

            cursor = collection.find(query).sort("created_at", 1)

            messages = []
            async for doc in cursor:
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
        await self._ensure_initialized()

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
            collection = await self._get_collection()
            await collection.insert_one(document)
        except Exception as e:
            logger.error(f"Error saving message for session {session_id}: {e}", exc_info=True)
            raise

    def setup(self) -> None:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.asetup())
