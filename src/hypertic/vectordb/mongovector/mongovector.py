import asyncio
import contextlib
from dataclasses import dataclass, field
from os import getenv
from typing import Any

from pymongo import AsyncMongoClient, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from hypertic.utils.log import get_logger
from hypertic.vectordb.base import BaseVectorDB, VectorDocument, VectorSearchResult

logger = get_logger(__name__)


@dataclass
class MongoDBAtlas(BaseVectorDB):
    collection: str
    embedder: Any | None = None
    connection_string: str | None = None
    database_name: str = "vectordb"
    vector_field: str = "embedding"

    collection_name: str = field(init=False)

    _client: MongoClient[Any] | None = field(default=None, init=False)
    _database: Database[Any] | None = field(default=None, init=False)
    _collection: Collection[Any] | None = field(default=None, init=False)
    _async_client: AsyncMongoClient[Any] | None = field(default=None, init=False)
    _async_database: Any = field(default=None, init=False)
    _async_collection: Collection[Any] | None = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self):
        BaseVectorDB.__init__(self, embedder=self.embedder)

        self.collection_name = self.collection

        self.connection_string = self.connection_string or getenv("MONGODB_VECTORDB_URL")

        if not self.connection_string:
            raise ValueError(
                "MongoDB connection string is required. Pass connection_string parameter or set MONGODB_VECTORDB_URL environment variable."
            )

    async def _maybe_await_async(self, result: Any) -> Any:
        if hasattr(result, "__await__"):
            return await result
        return result

    def _get_client(self) -> MongoClient[Any]:
        if self._client is None:
            self._client = MongoClient(self.connection_string)
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

    async def _get_async_client(self) -> AsyncMongoClient[Any]:
        if self._async_client is None:
            self._async_client = AsyncMongoClient(
                self.connection_string,
                maxPoolSize=100,
                retryWrites=True,
                serverSelectionTimeoutMS=5000,
            )
            await self._async_client.admin.command("ping")
            database_result = self._async_client[self.database_name]
            self._async_database = database_result
        return self._async_client

    async def _get_async_collection(self) -> Collection[Any]:
        if self._async_collection is None:
            await self._get_async_client()
            if self._async_database is None:
                raise RuntimeError("Async database not initialized")
            collection_result = self._async_database[self.collection_name]
            self._async_collection = collection_result
        return self._async_collection

    async def _initialize_db(self) -> bool:
        try:
            await self._get_async_client()
            collection = await self._get_async_collection()

            try:
                await self._maybe_await_async(collection.find_one())
            except Exception:
                await self._maybe_await_async(collection.insert_one({"_id": "__init_placeholder__"}))
                await self._maybe_await_async(collection.delete_one({"_id": "__init_placeholder__"}))
                await asyncio.sleep(0.1)

            # Try to create vector search index, but don't fail initialization if it fails
            try:
                await self._create_vector_search_index()
            except Exception as e:
                logger.debug(f"Vector search index creation failed (non-critical): {e}")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing MongoDB Atlas database: {e}", exc_info=True)
            return False

    def _initialize_db_sync(self) -> bool:
        try:
            client = self._get_client()
            client.admin.command("ping")

            self._get_database()
            collection = self._get_collection()

            try:
                collection.find_one()
            except Exception:
                collection.insert_one({"_id": "__placeholder__"})
                collection.delete_one({"_id": "__placeholder__"})

            self._create_vector_search_index_sync()

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing MongoDB Atlas database: {e}", exc_info=True)
            return False

    async def _create_vector_search_index(self):
        try:
            collection = await self._get_async_collection()

            try:
                await self._maybe_await_async(collection.find_one())
            except Exception as e:
                logger.debug(f"Collection not ready for index creation yet: {e}")
                return

            try:
                cursor_result = collection.list_search_indexes()
                cursor = await self._maybe_await_async(cursor_result)
                if cursor is None:
                    existing_indexes = []
                else:
                    to_list_result = cursor.to_list(None)
                    existing_indexes = await self._maybe_await_async(to_list_result)
                    if existing_indexes is None:
                        existing_indexes = []
            except Exception as e:
                logger.debug(f"Error listing search indexes (may not exist yet): {e}")
                existing_indexes = []

            index_exists = any(idx.get("name") == f"{self.collection_name}_vector_index" for idx in existing_indexes)

            if not index_exists:
                search_index_model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": self.vector_field,
                                "numDimensions": 1536,
                                "similarity": "cosine",
                            }
                        ]
                    },
                    name=f"{self.collection_name}_vector_index",
                    type="vectorSearch",
                )

                await self._maybe_await_async(collection.create_search_index(model=search_index_model))

        except Exception as e:
            error_msg = str(e).lower()
            if "does not exist" in error_msg or "namespace not found" in error_msg:
                logger.debug(f"Vector search index creation deferred (collection not ready): {e}")
            else:
                logger.error(f"Error creating vector search index: {e}", exc_info=True)

    def _create_vector_search_index_sync(self):
        try:
            collection = self._get_collection()

            try:
                collection.find_one()
            except Exception as e:
                logger.debug(f"Collection not ready for index creation yet: {e}")
                return

            try:
                existing_indexes = list(collection.list_search_indexes())
            except Exception:
                existing_indexes = []

            index_exists = any(idx.get("name") == f"{self.collection_name}_vector_index" for idx in existing_indexes)

            if not index_exists:
                search_index_model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": self.vector_field,
                                "numDimensions": 1536,
                                "similarity": "cosine",
                            }
                        ]
                    },
                    name=f"{self.collection_name}_vector_index",
                    type="vectorSearch",
                )

                collection.create_search_index(model=search_index_model)

        except Exception as e:
            error_msg = str(e).lower()
            if "does not exist" in error_msg or "namespace not found" in error_msg:
                logger.debug(f"Vector search index creation deferred (collection not ready): {e}")
            else:
                logger.error(f"Error creating vector search index: {e}", exc_info=True)

    async def _add_documents_impl(self, documents: list[VectorDocument]) -> bool:
        try:
            if not self._initialized:
                await self._initialize_db()

            collection = await self._get_async_collection()

            for doc in documents:
                if doc.vector is None:
                    logger.warning(f"Document {doc.id} has no vector, skipping")
                    continue

                doc_dict = {
                    "_id": doc.id,
                    self.vector_field: doc.vector,
                    "content": doc.content,
                    "metadata": doc.metadata or {},
                }

                await self._maybe_await_async(collection.replace_one({"_id": doc.id}, doc_dict, upsert=True))

            await self._create_vector_search_index()

            return True

        except Exception as e:
            logger.error(f"Error adding documents to MongoDB Atlas: {e}", exc_info=True)
            return False

    def _add_documents_impl_sync(self, documents: list[VectorDocument]) -> bool:
        try:
            if not self._initialized:
                self._initialize_db_sync()

            collection = self._get_collection()

            for doc in documents:
                if doc.vector is None:
                    logger.warning(f"Document {doc.id} has no vector, skipping")
                    continue

                doc_dict = {
                    "_id": doc.id,
                    self.vector_field: doc.vector,
                    "content": doc.content,
                    "metadata": doc.metadata or {},
                }

                collection.replace_one({"_id": doc.id}, doc_dict, upsert=True)

            self._create_vector_search_index_sync()

            return True

        except Exception as e:
            logger.error(f"Error adding documents to MongoDB Atlas: {e}", exc_info=True)
            return False

    async def _search_impl(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        try:
            if not self._initialized:
                await self._initialize_db()

            collection = await self._get_async_collection()

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": f"{self.collection_name}_vector_index",
                        "path": self.vector_field,
                        "queryVector": query_vector,
                        "numCandidates": top_k * 10,
                        "limit": top_k,
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "content": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]

            aggregate_result = collection.aggregate(pipeline)
            cursor = await self._maybe_await_async(aggregate_result)
            to_list_result = cursor.to_list(length=None)
            results = await self._maybe_await_async(to_list_result)

            search_results = []
            for result in results:
                search_results.append(
                    VectorSearchResult(
                        content=result.get("content", ""),
                        metadata=result.get("metadata", {}),
                        score=result.get("score", 0.0),
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Error searching MongoDB Atlas: {e}", exc_info=True)
            return []

    def _search_impl_sync(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        try:
            if not self._initialized:
                self._initialize_db_sync()

            collection = self._get_collection()

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": f"{self.collection_name}_vector_index",
                        "path": self.vector_field,
                        "queryVector": query_vector,
                        "numCandidates": top_k * 10,
                        "limit": top_k,
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "content": 1,
                        "metadata": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]

            results = list(collection.aggregate(pipeline))

            search_results = []
            for result in results:
                search_results.append(
                    VectorSearchResult(
                        content=result.get("content", ""),
                        metadata=result.get("metadata", {}),
                        score=result.get("score", 0.0),
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Error searching MongoDB Atlas: {e}", exc_info=True)
            return []

    async def _search_text_impl(self, query: str, top_k: int) -> list[VectorSearchResult]:
        if self.embedder:
            query_vector = await self.embedder.embed(query)
            return await self._search_impl(query_vector, top_k)
        else:
            return []

    def _search_text_impl_sync(self, query: str, top_k: int) -> list[VectorSearchResult]:
        if self.embedder:
            query_vector = self.embedder.embed_sync(query)
            return self._search_impl_sync(query_vector, top_k)
        else:
            return []

    async def _delete_impl(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        try:
            if not self._initialized:
                await self._initialize_db()

            collection = self._get_collection()

            if ids:
                result = await asyncio.to_thread(collection.delete_many, {"_id": {"$in": ids}})
            elif where:
                result = await asyncio.to_thread(collection.delete_many, where)
            else:
                logger.warning("No IDs or filter provided for deletion")
                return False

            return bool(result.deleted_count > 0)

        except Exception as e:
            logger.error(f"Error deleting documents from MongoDB Atlas: {e}", exc_info=True)
            return False

    def _delete_impl_sync(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        try:
            if not self._initialized:
                self._initialize_db_sync()

            collection = self._get_collection()

            if ids:
                result = collection.delete_many({"_id": {"$in": ids}})
            elif where:
                result = collection.delete_many(where)
            else:
                logger.warning("No IDs or filter provided for deletion")
                return False

            return bool(result.deleted_count > 0)

        except Exception as e:
            logger.error(f"Error deleting documents from MongoDB Atlas: {e}", exc_info=True)
            return False

    async def _update_impl(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        try:
            if not self._initialized:
                await self._initialize_db()

            collection = self._get_collection()

            for i, doc_id in enumerate(ids):
                update_doc: dict[str, Any] = {}

                if metadatas and i < len(metadatas):
                    update_doc["metadata"] = metadatas[i]

                if documents and i < len(documents):
                    update_doc["content"] = documents[i]

                if embeddings and i < len(embeddings):
                    update_doc[self.vector_field] = embeddings[i]

                if update_doc:
                    await asyncio.to_thread(collection.update_one, {"_id": doc_id}, {"$set": update_doc})

            return True

        except Exception as e:
            logger.error(f"Error updating documents in MongoDB Atlas: {e}", exc_info=True)
            return False

    def _update_impl_sync(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        try:
            if not self._initialized:
                self._initialize_db_sync()

            collection = self._get_collection()

            for i, doc_id in enumerate(ids):
                update_doc: dict[str, Any] = {}

                if metadatas and i < len(metadatas):
                    update_doc["metadata"] = metadatas[i]

                if documents and i < len(documents):
                    update_doc["content"] = documents[i]

                if embeddings and i < len(embeddings):
                    update_doc[self.vector_field] = embeddings[i]

                if update_doc:
                    collection.update_one({"_id": doc_id}, {"$set": update_doc})

            return True

        except Exception as e:
            logger.error(f"Error updating documents in MongoDB Atlas: {e}", exc_info=True)
            return False

    async def _upsert_impl(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        try:
            if not self._initialized:
                await self._initialize_db()

            collection = self._get_collection()

            for i, doc_id in enumerate(ids):
                doc_dict: dict[str, Any] = {"_id": doc_id}

                if metadatas and i < len(metadatas):
                    doc_dict["metadata"] = metadatas[i]

                content = documents[i] if documents and i < len(documents) else None
                if content:
                    doc_dict["content"] = content
                    if not embeddings or i >= len(embeddings):
                        if self.embedder:
                            doc_dict[self.vector_field] = await self.embedder.embed(content)
                    else:
                        doc_dict[self.vector_field] = embeddings[i]
                else:
                    if embeddings and i < len(embeddings):
                        doc_dict[self.vector_field] = embeddings[i]

                await asyncio.to_thread(collection.replace_one, {"_id": doc_id}, doc_dict, upsert=True)

            return True

        except Exception as e:
            logger.error(f"Error upserting documents in MongoDB Atlas: {e}", exc_info=True)
            return False

    def _upsert_impl_sync(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        try:
            if not self._initialized:
                self._initialize_db_sync()

            collection = self._get_collection()

            for i, doc_id in enumerate(ids):
                doc_dict: dict[str, Any] = {"_id": doc_id}

                if metadatas and i < len(metadatas):
                    doc_dict["metadata"] = metadatas[i]

                content = documents[i] if documents and i < len(documents) else None
                if content:
                    doc_dict["content"] = content
                    if not embeddings or i >= len(embeddings):
                        if self.embedder:
                            doc_dict[self.vector_field] = self.embedder.embed_sync(content)
                    else:
                        doc_dict[self.vector_field] = embeddings[i]
                else:
                    if embeddings and i < len(embeddings):
                        doc_dict[self.vector_field] = embeddings[i]

                collection.replace_one({"_id": doc_id}, doc_dict, upsert=True)

            return True

        except Exception as e:
            logger.error(f"Error upserting documents in MongoDB Atlas: {e}", exc_info=True)
            return False

    async def _get_documents_impl(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            if not self._initialized:
                await self._initialize_db()

            collection = await self._get_async_collection()

            query = {}
            if ids:
                query["_id"] = {"$in": ids}
            elif where:
                for key, value in where.items():
                    query[f"metadata.{key}"] = value

            projection = {"_id": 1, "content": 1, "metadata": 1}
            if include:
                actual_fields = []
                for field in include:
                    if field == "documents":
                        actual_fields.append("content")
                    elif field == "embeddings":
                        actual_fields.append(self.vector_field)
                    elif field == "ids":
                        actual_fields.append("_id")
                    else:
                        actual_fields.append(field)
                projection = dict.fromkeys(actual_fields, 1)

            cursor = collection.find(query, projection)
            if limit:
                cursor = cursor.limit(limit)

            to_list_result = cursor.to_list(length=None)
            results = await self._maybe_await_async(to_list_result)

            result_ids = [doc.get("_id", "") for doc in results]
            contents = [doc.get("content", "") for doc in results]
            metadatas = [doc.get("metadata", {}) for doc in results]
            vectors = [doc.get(self.vector_field, []) for doc in results] if include and "embeddings" in include else []

            return_dict = {"ids": result_ids}
            if include:
                if "documents" in include:
                    return_dict["documents"] = contents
                if "contents" in include:
                    return_dict["contents"] = contents
                if "metadatas" in include:
                    return_dict["metadatas"] = metadatas
                if "embeddings" in include:
                    return_dict["embeddings"] = vectors
            else:
                return_dict.update({"documents": contents, "contents": contents, "metadatas": metadatas})

            return return_dict

        except Exception as e:
            logger.error(f"Error getting documents from MongoDB Atlas: {e}", exc_info=True)
            return {"ids": [], "contents": [], "metadatas": []}

    def _get_documents_impl_sync(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            if not self._initialized:
                self._initialize_db_sync()

            collection = self._get_collection()

            query = {}
            if ids:
                query["_id"] = {"$in": ids}
            elif where:
                for key, value in where.items():
                    query[f"metadata.{key}"] = value

            projection = {"_id": 1, "content": 1, "metadata": 1}
            if include:
                actual_fields = []
                for field in include:
                    if field == "documents":
                        actual_fields.append("content")
                    elif field == "embeddings":
                        actual_fields.append(self.vector_field)
                    elif field == "ids":
                        actual_fields.append("_id")
                    else:
                        actual_fields.append(field)
                projection = dict.fromkeys(actual_fields, 1)

            cursor = collection.find(query, projection)
            if limit:
                cursor = cursor.limit(limit)

            results = list(cursor)

            result_ids = [doc.get("_id", "") for doc in results]
            contents = [doc.get("content", "") for doc in results]
            metadatas = [doc.get("metadata", {}) for doc in results]
            vectors = [doc.get(self.vector_field, []) for doc in results] if include and "embeddings" in include else []

            return_dict = {"ids": result_ids}
            if include:
                if "documents" in include:
                    return_dict["documents"] = contents
                if "contents" in include:
                    return_dict["contents"] = contents
                if "metadatas" in include:
                    return_dict["metadatas"] = metadatas
                if "embeddings" in include:
                    return_dict["embeddings"] = vectors
            else:
                return_dict.update({"documents": contents, "contents": contents, "metadatas": metadatas})

            return return_dict

        except Exception as e:
            logger.error(f"Error getting documents from MongoDB Atlas: {e}", exc_info=True)
            return {"ids": [], "contents": [], "metadatas": []}

    async def _count_documents_impl(self) -> int:
        try:
            if not self._initialized:
                await self._initialize_db()

            collection = await self._get_async_collection()
            count_result = collection.count_documents({})
            count = await self._maybe_await_async(count_result)
            return int(count) if count is not None else 0

        except Exception as e:
            logger.error(f"Error counting documents in MongoDB Atlas: {e}", exc_info=True)
            return 0

    def _count_documents_impl_sync(self) -> int:
        try:
            if not self._initialized:
                self._initialize_db_sync()

            collection = self._get_collection()
            count = collection.count_documents({})
            return int(count)

        except Exception as e:
            logger.error(f"Error counting documents in MongoDB Atlas: {e}", exc_info=True)
            return 0

    def exists(self) -> bool:
        try:
            if not self._initialized:
                with contextlib.suppress(Exception):
                    self._initialize_db_sync()

            collection = self._get_collection()
            collection.database.list_collection_names()
            collections = collection.database.list_collection_names()
            return self.collection_name in collections

        except Exception:
            return False
