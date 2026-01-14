from importlib import import_module
from typing import TYPE_CHECKING

from hypertic.vectordb.base import BaseVectorDB, VectorDocument, VectorSearchResult

if TYPE_CHECKING:
    from hypertic.vectordb.chroma.chroma import ChromaDB
    from hypertic.vectordb.mongovector.mongovector import MongoDBAtlas
    from hypertic.vectordb.pgvector.pgvector import PgVectorDB
    from hypertic.vectordb.pinecone.pinecone import PineconeDB
    from hypertic.vectordb.qdrant.qdrant import QdrantDB

_module_lookup = {
    "ChromaDB": "hypertic.vectordb.chroma.chroma",
    "MongoDBAtlas": "hypertic.vectordb.mongovector.mongovector",
    "PgVectorDB": "hypertic.vectordb.pgvector.pgvector",
    "PineconeDB": "hypertic.vectordb.pinecone.pinecone",
    "QdrantDB": "hypertic.vectordb.qdrant.qdrant",
}


def __getattr__(name: str):
    if name in _module_lookup:
        module_path = _module_lookup[name]
        module = import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_module_lookup.keys()) + ["BaseVectorDB", "VectorDocument", "VectorSearchResult"]
