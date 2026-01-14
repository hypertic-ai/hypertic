import asyncio
import json
from contextlib import asynccontextmanager
from os import getenv
from typing import Any

from hypertic.memory.base import BaseMemory
from hypertic.utils.log import get_logger

logger = get_logger(__name__)

try:
    from sqlalchemy import Index, inspect, select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from hypertic.memory.postgres.postgres import MemoryStore, SQLAlchemyBase
except ImportError as err:
    raise ImportError(
        "SQLAlchemy 2.0+ and greenlet required for AsyncPostgresServer. Install with: pip install 'sqlalchemy[asyncio]>=2.0.0' greenlet>=3.0.0"
    ) from err


class AsyncPostgresServer(BaseMemory):
    def __init__(self, db_url: str | None = None, table_name: str = "agent_memory"):
        db_url = db_url or getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("db_url is required. Set DATABASE_URL environment variable or pass db_url parameter.")

        if not db_url.startswith("postgresql+asyncpg://"):
            if db_url.startswith("postgresql://"):
                db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            else:
                raise ValueError(
                    "db_url must use asyncpg driver. Use 'postgresql+asyncpg://...' or 'postgresql://...' (will be converted automatically)"
                )

        self.db_url = db_url
        self.table_name = table_name
        self.engine = create_async_engine(db_url, echo=False)
        self.AsyncSession = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        self._initialized = False
        self._lock = asyncio.Lock()
        MemoryStore.__table__.name = table_name
        self._setup_indexes()

    def _setup_indexes(self) -> None:
        idx_prefix = f"idx_{self.table_name}"
        MemoryStore.__table__.indexes.clear()
        MemoryStore.__table__.append_constraint(Index(f"{idx_prefix}_session_created", MemoryStore.session_id, MemoryStore.created_at))
        MemoryStore.__table__.append_constraint(Index(f"{idx_prefix}_user_id", MemoryStore.user_id))
        MemoryStore.__table__.append_constraint(Index(f"{idx_prefix}_session_id", MemoryStore.session_id))

    @classmethod
    @asynccontextmanager
    async def create(cls, db_url: str, *, table_name: str = "agent_memory", **engine_kwargs):
        instance = cls(db_url=db_url, table_name=table_name)
        try:
            yield instance
        finally:
            await instance.engine.dispose()

    async def asetup(self) -> None:
        async with self._lock:
            if self._initialized:
                return

            def check_table_exists(conn):
                inspector = inspect(conn)
                return inspector.has_table(self.table_name)

            async with self.engine.begin() as conn:
                table_exists = await conn.run_sync(check_table_exists)

                if table_exists:
                    logger.debug(f"Table {self.table_name} already exists")
                    self._initialized = True
                    return

            table_was_created = False

            try:
                async with self.engine.begin() as conn:
                    await conn.run_sync(SQLAlchemyBase.metadata.create_all)
                    table_was_created = True
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    async with self.engine.begin() as conn:
                        table_exists_after = await conn.run_sync(check_table_exists)
                        if table_exists_after:
                            logger.debug(f"Table {self.table_name} exists (index conflict ignored)")
                        else:
                            logger.warning(f"Index conflict but table {self.table_name} doesn't exist. This may indicate a database state issue.")
                            raise
                else:
                    raise

            async with self.engine.begin() as conn:
                table_exists_final = await conn.run_sync(check_table_exists)
                if not table_exists_final:
                    raise RuntimeError(f"Table {self.table_name} was not created successfully")

            if table_was_created:
                logger.info(f"AsyncPostgresServer table ({self.table_name}) created successfully")

            self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.asetup()

    async def aget_messages(self, session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        await self._ensure_initialized()

        try:
            async with self.AsyncSession() as session:
                stmt = select(MemoryStore)

                if session_id:
                    stmt = stmt.where(MemoryStore.session_id == session_id)
                elif user_id:
                    stmt = stmt.where(MemoryStore.user_id == user_id)
                else:
                    return []

                stmt = stmt.order_by(MemoryStore.created_at.asc())

                result = await session.execute(stmt)
                results = result.scalars().all()

                messages = []
                for row in results:
                    message_data: dict[str, Any] | Any = row.message
                    if not isinstance(message_data, dict):
                        try:
                            parsed_data: dict[str, Any] = json.loads(str(message_data)) if message_data else {}
                            message_data = parsed_data
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"Invalid message format for id {row.id}, skipping")
                            continue

                    if message_data.get("type") == "user_data":
                        continue

                    if "role" not in message_data or "content" not in message_data:
                        logger.warning(f"Invalid message format for id {row.id}, skipping")
                        continue

                    message = {
                        "role": message_data.get("role"),
                        "content": message_data.get("content"),
                        "session_id": row.session_id,
                        "created_at": row.created_at,
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

        try:
            async with self.AsyncSession() as session:
                memory_item = MemoryStore(
                    session_id=session_id,
                    user_id=user_id,
                    message=message_data,
                    tool_calls=tool_calls,
                    tool_outputs=tool_outputs,
                    response_metadata=metadata,
                )
                session.add(memory_item)
                await session.commit()
        except Exception as e:
            logger.error(f"Error saving message for session {session_id}: {e}", exc_info=True)
            raise

    def get_messages(self, session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aget_messages(session_id, user_id))
        else:
            raise RuntimeError("Cannot call get_messages() from async context. Use aget_messages() instead.")

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
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.asave_message(session_id, role, content, user_id, tool_calls, tool_outputs, metadata))
        else:
            raise RuntimeError("Cannot call save_message() from async context. Use asave_message() instead.")

    def setup(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.asetup())
        else:
            raise RuntimeError("Cannot call setup() from async context. Use asetup() instead.")
