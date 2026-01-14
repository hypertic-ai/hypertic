import hashlib
import json
from os import getenv
from typing import Any

from hypertic.memory.base import BaseMemory
from hypertic.utils.log import get_logger, mask_connection_string

logger = get_logger(__name__)

try:
    from redis.asyncio import Redis, from_url
    from redis.exceptions import RedisError
except ImportError as err:
    raise ImportError("redis[asyncio] required for AsyncRedisCache. Install with: pip install 'redis[asyncio]'") from err


class AsyncRedisCache(BaseMemory):
    def __init__(
        self,
        store: BaseMemory,
        redis_url: str | None = None,
        ttl: int = 3600,
        key_prefix: str = "cache:memory:",
    ):
        self.store = store
        self.ttl = ttl
        self.key_prefix = key_prefix
        self.redis_url = redis_url or getenv("REDIS_URL") or "redis://localhost:6379"
        self.redis_client: Redis | None = None
        self._initialized = False

    async def _ensure_redis_connected(self) -> None:
        if not self._initialized:
            try:
                self.redis_client = await from_url(self.redis_url, decode_responses=True)
                if self.redis_client is not None:
                    ping_result = self.redis_client.ping()
                    if hasattr(ping_result, "__await__"):
                        await ping_result
                logger.info(f"AsyncRedis cache connected: {mask_connection_string(self.redis_url)}")
                self._initialized = True
            except RedisError as e:
                logger.warning(f"Redis connection failed: {e}. Cache will be disabled.")
                self.redis_client = None
                self._initialized = True

    async def _close_redis(self) -> None:
        if self.redis_client:
            try:
                await self.redis_client.aclose()
                logger.debug("AsyncRedis connection closed")
            except RedisError as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self.redis_client = None
                self._initialized = False

    def _make_cache_key(self, session_id: str | None = None, user_id: str | None = None) -> str | None:
        key_parts = []
        if session_id:
            key_parts.append(f"session:{session_id}")
        if user_id:
            key_parts.append(f"user:{user_id}")

        if not key_parts:
            return None

        key_string = "|".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{self.key_prefix}{key_hash}"

    def get_messages(self, session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        return self.store.get_messages(session_id=session_id, user_id=user_id)

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
        self.store.save_message(
            session_id=session_id,
            role=role,
            content=content,
            user_id=user_id,
            tool_calls=tool_calls,
            tool_outputs=tool_outputs,
            metadata=metadata,
        )

    async def aget_messages(self, session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        await self._ensure_redis_connected()

        cache_key = None

        if self.redis_client:
            cache_key = self._make_cache_key(session_id, user_id)
            if cache_key:
                try:
                    cached = await self.redis_client.get(cache_key)
                    if cached:
                        loaded_data: Any = json.loads(cached)
                        messages: list[dict[str, Any]] = loaded_data if isinstance(loaded_data, list) else []
                        logger.debug("Cache HIT: Using cached messages from Redis")
                        return messages
                    logger.debug("Cache MISS: Fetching from store (PostgreSQL/MongoDB)")
                except (RedisError, json.JSONDecodeError) as e:
                    logger.warning(f"Cache read error: {e}. Falling back to store.")

        messages = await self.store.aget_messages(session_id=session_id, user_id=user_id)

        cleaned_messages = []
        for msg in messages:
            cleaned_msg = {k: v for k, v in msg.items() if k not in ("session_id", "created_at", "updated_at")}
            cleaned_messages.append(cleaned_msg)

        if self.redis_client and cache_key:
            try:
                await self.redis_client.setex(cache_key, self.ttl, json.dumps(cleaned_messages))
                logger.debug(f"Cache SET: Stored messages in Redis (TTL: {self.ttl}s)")
            except (RedisError, TypeError) as e:
                logger.warning(f"Cache write error: {e}")

        return messages

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
        await self._ensure_redis_connected()

        await self.store.asave_message(
            session_id=session_id,
            role=role,
            content=content,
            user_id=user_id,
            tool_calls=tool_calls,
            tool_outputs=tool_outputs,
            metadata=metadata,
        )

        if self.redis_client:
            cache_key = self._make_cache_key(session_id, user_id)
            if cache_key:
                try:
                    messages = await self.store.aget_messages(session_id=session_id, user_id=user_id)

                    cleaned_messages = []
                    for msg in messages:
                        cleaned_msg = {k: v for k, v in msg.items() if k not in ("session_id", "created_at", "updated_at")}
                        cleaned_messages.append(cleaned_msg)

                    await self.redis_client.setex(cache_key, self.ttl, json.dumps(cleaned_messages))
                    logger.debug("Cache UPDATED: Updated messages in Redis")
                except (RedisError, TypeError) as e:
                    logger.warning(f"Cache update error: {e}")

    def setup(self) -> None:
        self.store.setup()

    async def asetup(self) -> None:
        await self.store.asetup()

    async def __aenter__(self):
        await self._ensure_redis_connected()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close_redis()
