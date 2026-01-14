import asyncio
import hashlib
import json
from os import getenv
from typing import Any

from hypertic.memory.base import BaseMemory
from hypertic.utils.log import get_logger, mask_connection_string

logger = get_logger(__name__)

try:
    import redis
    from redis.exceptions import RedisError
except ImportError as err:
    raise ImportError("redis required for RedisCache. Install with: pip install redis") from err


class RedisCache(BaseMemory):
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

        redis_url = redis_url or getenv("REDIS_URL") or "redis://localhost:6379"

        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"Redis cache connected: {mask_connection_string(redis_url)}")
        except RedisError as e:
            logger.warning(f"Redis connection failed: {e}. Cache will be disabled.")
            self.redis_client = None

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
        if self.redis_client:
            cache_key = self._make_cache_key(session_id, user_id)
            if cache_key:
                try:
                    cached = self.redis_client.get(cache_key)
                    if cached:
                        loaded_data: Any = json.loads(cached)
                        messages: list[dict[str, Any]] = loaded_data if isinstance(loaded_data, list) else []
                        logger.debug("Cache HIT: Using cached messages from Redis")
                        return messages
                    logger.debug("Cache MISS: Fetching from store (PostgreSQL/MongoDB)")
                except (RedisError, json.JSONDecodeError) as e:
                    logger.warning(f"Cache read error: {e}. Falling back to store.")

        messages = self.store.get_messages(session_id=session_id, user_id=user_id)

        cleaned_messages = []
        for msg in messages:
            cleaned_msg = {k: v for k, v in msg.items() if k not in ("session_id", "created_at", "updated_at")}
            cleaned_messages.append(cleaned_msg)

        if self.redis_client and cache_key:
            try:
                self.redis_client.setex(cache_key, self.ttl, json.dumps(cleaned_messages))
                logger.debug(f"Cache SET: Stored messages in Redis (TTL: {self.ttl}s)")
            except (RedisError, TypeError) as e:
                logger.warning(f"Cache write error: {e}")

        return messages

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

        if self.redis_client:
            cache_key = self._make_cache_key(session_id, user_id)
            if cache_key:
                try:
                    messages = self.store.get_messages(session_id=session_id, user_id=user_id)

                    cleaned_messages = []
                    for msg in messages:
                        cleaned_msg = {k: v for k, v in msg.items() if k not in ("session_id", "created_at", "updated_at")}
                        cleaned_messages.append(cleaned_msg)

                    self.redis_client.setex(cache_key, self.ttl, json.dumps(cleaned_messages))
                    logger.debug("Cache UPDATED: Updated messages in Redis")
                except (RedisError, TypeError) as e:
                    logger.warning(f"Cache update error: {e}")

    async def aget_messages(self, session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        import asyncio

        cache_key = None

        if self.redis_client:
            cache_key = self._make_cache_key(session_id, user_id)
            if cache_key:
                try:
                    loop = asyncio.get_event_loop()
                    cached = await loop.run_in_executor(None, self.redis_client.get, cache_key)
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
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.redis_client.setex(cache_key, self.ttl, json.dumps(cleaned_messages)),
                )
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

                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: self.redis_client.setex(cache_key, self.ttl, json.dumps(cleaned_messages)),
                    )
                    logger.debug("Cache UPDATED: Updated messages in Redis")
                except (RedisError, TypeError) as e:
                    logger.warning(f"Cache update error: {e}")

    def setup(self) -> None:
        self.store.setup()

    async def asetup(self) -> None:
        await self.store.asetup()
