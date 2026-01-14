import json
from datetime import datetime, timezone
from os import getenv
from typing import Any

from hypertic.memory.base import BaseMemory
from hypertic.utils.log import get_logger

logger = get_logger(__name__)

try:
    from sqlalchemy import (
        Column,
        DateTime,
        Index,
        Integer,
        String,
        create_engine,
        inspect,
        select,
    )
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
except ImportError as err:
    raise ImportError("SQLAlchemy 2.0+ required for PostgresServer. Install with: pip install 'sqlalchemy>=2.0.0'") from err

SQLAlchemyBase = declarative_base()


class MemoryStore(SQLAlchemyBase):  # type: ignore
    __tablename__ = "agent_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)

    session_id = Column(String(255), nullable=False)
    user_id = Column(String(255), nullable=True)
    message = Column(JSONB, nullable=False)
    tool_calls = Column(JSONB, nullable=True)
    tool_outputs = Column(JSONB, nullable=True)
    response_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        nullable=False,
    )


class PostgresServer(BaseMemory):
    def __init__(self, db_url: str | None = None, table_name: str = "agent_memory"):
        self.db_url = db_url or getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("db_url is required. Set DATABASE_URL environment variable or pass db_url parameter.")
        self.table_name = table_name
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self._initialized = False
        MemoryStore.__table__.name = table_name
        self._setup_indexes()

    def _setup_indexes(self) -> None:
        idx_prefix = f"idx_{self.table_name}"
        MemoryStore.__table__.indexes.clear()
        MemoryStore.__table__.append_constraint(Index(f"{idx_prefix}_session_created", MemoryStore.session_id, MemoryStore.created_at))
        MemoryStore.__table__.append_constraint(Index(f"{idx_prefix}_user_id", MemoryStore.user_id))
        MemoryStore.__table__.append_constraint(Index(f"{idx_prefix}_session_id", MemoryStore.session_id))

    def setup(self) -> None:
        if not self._initialized:
            inspector = inspect(self.engine)
            table_exists = inspector.has_table(self.table_name)

            try:
                SQLAlchemyBase.metadata.create_all(self.engine)
            except Exception as e:
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    try:
                        with self.Session() as session:
                            session.execute(select(MemoryStore).limit(1)).first()
                        logger.debug(f"Table {self.table_name} exists (index conflict ignored)")
                    except Exception:
                        logger.error(f"Index conflict but table {self.table_name} doesn't exist. This may indicate a database state issue.")
                        raise
                else:
                    raise

            if not table_exists:
                logger.info(f"PostgresServer table ({self.table_name}) created successfully")

            self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.setup()

    def get_messages(self, session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        self._ensure_initialized()

        try:
            with self.Session() as session:
                stmt = select(MemoryStore)

                if session_id:
                    stmt = stmt.where(MemoryStore.session_id == session_id)
                elif user_id:
                    stmt = stmt.where(MemoryStore.user_id == user_id)
                else:
                    return []

                stmt = stmt.order_by(MemoryStore.created_at.asc())

                result = session.execute(stmt)
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

        try:
            with self.Session() as session:
                memory_item = MemoryStore(
                    session_id=session_id,
                    user_id=user_id,
                    message=message_data,
                    tool_calls=tool_calls,
                    tool_outputs=tool_outputs,
                    response_metadata=metadata,
                )
                session.add(memory_item)
                session.commit()
        except Exception as e:
            logger.error(f"Error saving message for session {session_id}: {e}", exc_info=True)
            raise
