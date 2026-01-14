from importlib import import_module
from typing import TYPE_CHECKING

from hypertic.memory.base import BaseMemory
from hypertic.memory.inmemory import InMemory

if TYPE_CHECKING:
    from hypertic.memory.mongodb.mongo import MongoServer
    from hypertic.memory.mongodb.mongo_async import AsyncMongoServer
    from hypertic.memory.postgres.postgres import PostgresServer
    from hypertic.memory.postgres.postgres_async import AsyncPostgresServer
    from hypertic.memory.redis.async_cache import AsyncRedisCache
    from hypertic.memory.redis.cache import RedisCache

_module_lookup = {
    "PostgresServer": "hypertic.memory.postgres.postgres",
    "AsyncPostgresServer": "hypertic.memory.postgres.postgres_async",
    "MongoServer": "hypertic.memory.mongodb.mongo",
    "AsyncMongoServer": "hypertic.memory.mongodb.mongo_async",
    "RedisCache": "hypertic.memory.redis.cache",
    "AsyncRedisCache": "hypertic.memory.redis.async_cache",
}


def __getattr__(name: str):
    if name in _module_lookup:
        module_path = _module_lookup[name]
        module = import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_module_lookup.keys()) + ["BaseMemory", "InMemory"]
