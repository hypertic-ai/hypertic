import hashlib
import json
from dataclasses import dataclass, field
from os import getenv
from typing import Any
from uuid import UUID

from hypertic.utils.log import get_logger
from hypertic.vectordb.base import BaseVectorDB, VectorDocument, VectorSearchResult

logger = get_logger(__name__)

try:
    from qdrant_client import AsyncQdrantClient, QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        FilterSelector,
        MatchAny,
        MatchValue,
        PointIdsList,
        PointStruct,
        Range,
        ScoredPoint,
        VectorParams,
    )
except ImportError as err:
    raise ImportError("`qdrant-client` not installed. Install with: pip install qdrant-client") from err


@dataclass
class QdrantDB(BaseVectorDB):
    collection: str
    embedder: Any | None = None
    host: str | None = None
    port: int = 6333
    grpc_port: int = 6334
    url: str | None = None
    path: str | None = None
    api_key: str | None = None
    prefer_grpc: bool = False
    timeout: int | None = None
    vector_size: int | None = None
    distance_metric: str = "Cosine"

    collection_name: str = field(init=False)

    _client: QdrantClient | None = field(default=None, init=False)
    _async_client: AsyncQdrantClient | None = field(default=None, init=False)

    def __post_init__(self):
        BaseVectorDB.__init__(self, embedder=self.embedder)

        self.collection_name = self.collection

    def _get_cloud_config(self) -> str | None:
        return self.api_key or getenv("QDRANT_API_KEY")

    def _get_distance_metric(self) -> Distance:
        distance_map = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclidean": Distance.EUCLID,
            "Manhattan": Distance.MANHATTAN,
        }
        return distance_map.get(self.distance_metric, Distance.COSINE)

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            api_key = self._get_cloud_config()

            if api_key or self.api_key:
                if self.url:
                    if api_key is not None:
                        self._client = QdrantClient(url=self.url, api_key=api_key, timeout=self.timeout)
                    else:
                        self._client = QdrantClient(url=self.url, timeout=self.timeout)
                else:
                    if self.path:
                        self._client = QdrantClient(path=self.path)
                    else:
                        self._client = QdrantClient(":memory:")
            elif self.url:
                self._client = QdrantClient(url=self.url, timeout=self.timeout)
            elif self.host:
                self._client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    grpc_port=self.grpc_port,
                    prefer_grpc=self.prefer_grpc,
                    timeout=self.timeout,
                )
            elif self.path:
                self._client = QdrantClient(path=self.path)
            else:
                self._client = QdrantClient(":memory:")

        return self._client

    async def _get_async_client(self) -> AsyncQdrantClient:
        if self._async_client is None:
            api_key = self._get_cloud_config()

            if api_key or self.api_key:
                if self.url:
                    if api_key is not None:
                        self._async_client = AsyncQdrantClient(url=self.url, api_key=api_key, timeout=self.timeout)
                    else:
                        self._async_client = AsyncQdrantClient(url=self.url, timeout=self.timeout)
                else:
                    if self.path:
                        self._async_client = AsyncQdrantClient(path=self.path)
                    else:
                        self._async_client = AsyncQdrantClient(":memory:")
            elif self.url:
                self._async_client = AsyncQdrantClient(url=self.url, timeout=self.timeout)
            elif self.host:
                self._async_client = AsyncQdrantClient(
                    host=self.host,
                    port=self.port,
                    grpc_port=self.grpc_port,
                    prefer_grpc=self.prefer_grpc,
                    timeout=self.timeout,
                )
            elif self.path:
                self._async_client = AsyncQdrantClient(path=self.path)
            else:
                self._async_client = AsyncQdrantClient(":memory:")

        return self._async_client

    def _flatten_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        flattened: dict[str, Any] = {}

        def _flatten_recursive(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                if len(obj) == 0:
                    flattened[prefix] = json.dumps(obj)
                else:
                    for key, value in obj.items():
                        new_key = f"{prefix}.{key}" if prefix else key
                        _flatten_recursive(value, new_key)
            elif isinstance(obj, list | tuple):
                flattened[prefix] = json.dumps(obj)
            elif isinstance(obj, str | int | float | bool) or obj is None:
                if obj is not None:
                    flattened[prefix] = obj
            else:
                try:
                    flattened[prefix] = json.dumps(obj)
                except (TypeError, ValueError):
                    flattened[prefix] = str(obj)

        _flatten_recursive(metadata)
        return flattened

    def _create_filter(self, where: dict[str, Any]) -> Filter | None:
        """Create Qdrant filter from where clause"""
        if not where:
            return None

        conditions = []

        for key, value in where.items():
            if isinstance(value, dict):
                if "gte" in value or "gt" in value or "lte" in value or "lt" in value:
                    range_conditions = {}
                    if "gte" in value:
                        range_conditions["gte"] = value["gte"]
                    if "gt" in value:
                        range_conditions["gt"] = value["gt"]
                    if "lte" in value:
                        range_conditions["lte"] = value["lte"]
                    if "lt" in value:
                        range_conditions["lt"] = value["lt"]

                    conditions.append(FieldCondition(key=key, range=Range(**range_conditions)))
                elif "in" in value:
                    match_values = value["in"]
                    if not isinstance(match_values, list):
                        match_values = [match_values]
                    match_obj = MatchAny(any=match_values)
                    conditions.append(FieldCondition(key=key, match=match_obj))
                elif "nin" in value:
                    for val in value["nin"]:
                        conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        if not conditions:
            return None
        must_list: list[Any] = conditions
        return Filter(must=must_list)

    async def _initialize_db(self) -> bool:
        try:
            client = await self._get_async_client()

            vector_size = self.vector_size
            if not vector_size and self.embedder:
                if hasattr(self.embedder, "dimensions"):
                    vector_size = self.embedder.dimensions
                elif hasattr(self.embedder, "model_dimensions"):
                    vector_size = self.embedder.model_dimensions
                elif hasattr(self.embedder, "get_embedding_dimension"):
                    vector_size = self.embedder.get_embedding_dimension()
                else:
                    try:
                        if hasattr(self.embedder, "initialize_sync"):
                            self.embedder.initialize_sync()
                        if hasattr(self.embedder, "embed_sync"):
                            test_embedding = self.embedder.embed_sync("test")
                            if test_embedding and len(test_embedding) > 0:
                                vector_size = len(test_embedding)
                                logger.info(f"Detected embedder dimension: {vector_size}")
                    except Exception as e:
                        logger.warning(f"Could not detect embedder dimension: {e}")
                    if not vector_size:
                        vector_size = 1536
                        logger.warning(f"Using default vector size: {vector_size}")

            if not vector_size:
                raise ValueError("Vector size must be specified either directly or through embedder")

            try:
                existing_collection = await client.get_collection(self.collection_name)
                vectors_config = existing_collection.config.params.vectors
                if vectors_config is None:
                    raise ValueError("Collection has no vector configuration")
                if isinstance(vectors_config, dict):
                    if not vectors_config:
                        raise ValueError("Collection has empty vector configuration")
                    existing_dim = next(iter(vectors_config.values())).size
                else:
                    existing_dim = vectors_config.size
                if existing_dim != vector_size:
                    logger.warning(
                        f"Collection '{self.collection_name}' exists with dimension {existing_dim}, "
                        f"but embedder requires {vector_size}. Deleting and recreating collection."
                    )
                    await client.delete_collection(self.collection_name)
                    await client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=self._get_distance_metric()),
                    )
            except Exception:
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=self._get_distance_metric()),
                )

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}", exc_info=True)
            return False

    def exists(self) -> bool:
        try:
            client = self._get_client()
            return bool(client.collection_exists(self.collection_name))
        except Exception:
            return False

    async def async_exists(self) -> bool:
        try:
            client = await self._get_async_client()
            return bool(await client.collection_exists(self.collection_name))
        except Exception:
            return False

    async def _get_documents_impl(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            if not await self.async_exists():
                await self._initialize_db()
            client = await self._get_async_client()

            if ids:
                points = await client.retrieve(
                    collection_name=self.collection_name,
                    ids=ids,
                    with_payload=True,
                    with_vectors=True,
                )

                result = {
                    "ids": [point.id for point in points],
                    "metadatas": [point.payload or {} for point in points],
                    "documents": [(point.payload or {}).get("content", "") for point in points],
                    "embeddings": [point.vector for point in points] if points and points[0].vector else [],
                }
            else:
                filter_condition = self._create_filter(where) if where else None

                scroll_result = await client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    limit=limit or 10000,
                    with_payload=True,
                    with_vectors=True,
                )

                points = scroll_result[0]

                result = {
                    "ids": [point.id for point in points],
                    "metadatas": [point.payload or {} for point in points],
                    "documents": [(point.payload or {}).get("content", "") for point in points],
                    "embeddings": [point.vector for point in points] if points and points[0].vector else [],
                }

            return result

        except Exception as e:
            logger.error(f"Error getting documents: {e}", exc_info=True)
            return {"ids": [], "metadatas": [], "documents": [], "embeddings": []}

    async def _count_documents_impl(self) -> int:
        try:
            if not await self.async_exists():
                await self._initialize_db()
            client = await self._get_async_client()
            collection_info = await client.get_collection(self.collection_name)
            count = collection_info.points_count
            return int(count) if count is not None else 0
        except Exception as e:
            logger.error(f"Error counting documents: {e}", exc_info=True)
            return 0

    def _get_documents_impl_sync(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            if not self.exists():
                self._initialize_db_sync()
            client = self._get_client()

            if ids:
                points = client.retrieve(
                    collection_name=self.collection_name,
                    ids=ids,
                    with_payload=True,
                    with_vectors=True,
                )

                result = {
                    "ids": [point.id for point in points],
                    "metadatas": [point.payload or {} for point in points],
                    "documents": [(point.payload or {}).get("content", "") for point in points],
                    "embeddings": [point.vector for point in points] if points and points[0].vector else [],
                }
            else:
                filter_condition = self._create_filter(where) if where else None

                scroll_result = client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    limit=limit or 10000,
                    with_payload=True,
                    with_vectors=True,
                )

                points = scroll_result[0]

                result = {
                    "ids": [point.id for point in points],
                    "metadatas": [point.payload or {} for point in points],
                    "documents": [(point.payload or {}).get("content", "") for point in points],
                    "embeddings": [point.vector for point in points] if points and points[0].vector else [],
                }

            return result

        except Exception as e:
            logger.error(f"Error getting documents: {e}", exc_info=True)
            return {"ids": [], "metadatas": [], "documents": [], "embeddings": []}

    def _count_documents_impl_sync(self) -> int:
        try:
            if not self.exists():
                self._initialize_db_sync()
            client = self._get_client()
            collection_info = client.get_collection(self.collection_name)
            count = collection_info.points_count
            return int(count) if count is not None else 0
        except Exception as e:
            logger.error(f"Error counting documents: {e}", exc_info=True)
            return 0

    async def _add_documents_impl(self, documents: list[VectorDocument]) -> bool:
        try:
            if not await self.async_exists():
                await self._initialize_db()

            client = await self._get_async_client()

            points = []
            for document in documents:
                cleaned_content = document.content.replace("\x00", "\ufffd")

                if not document.id:
                    doc_id = hashlib.md5(cleaned_content.encode()).hexdigest()
                else:
                    doc_id = document.id

                try:
                    existing_points = await client.retrieve(
                        collection_name=self.collection_name,
                        ids=[doc_id],
                        with_payload=False,
                        with_vectors=False,
                    )
                    if existing_points:
                        continue
                except Exception:
                    pass

                metadata = document.metadata or {}
                flattened_metadata = self._flatten_metadata(metadata)

                payload = {"content": cleaned_content, **flattened_metadata}

                if self.embedder and document.vector:
                    vector = document.vector
                elif self.embedder:
                    vector = await self.embedder.embed(cleaned_content)
                elif document.vector:
                    vector = document.vector
                else:
                    if self.vector_size is None:
                        raise ValueError("vector_size must be specified when no embedder is provided")
                    vector = [0.0] * self.vector_size

                if vector is None:
                    raise ValueError("Vector cannot be None for PointStruct")
                point = PointStruct(id=doc_id, vector=vector, payload=payload)
                points.append(point)

            if points:
                await client.upsert(collection_name=self.collection_name, points=points)

            return True

        except Exception as e:
            logger.error(f"Error adding documents to Qdrant: {e}", exc_info=True)
            return False

    async def _search_impl(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        try:
            if not await self.async_exists():
                await self._initialize_db()

            client = await self._get_async_client()

            query_response = await client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )

            return self._format_search_results(query_response.points)

        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}", exc_info=True)
            return []

    async def _search_text_impl(self, query: str, top_k: int) -> list[VectorSearchResult]:
        try:
            if not await self.async_exists():
                await self._initialize_db()

            if not self.embedder:
                raise ValueError("Embedder required for text search")

            query_vector = await self.embedder.embed(query)
            return await self._search_impl(query_vector, top_k)

        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}", exc_info=True)
            return []

    def _format_search_results(self, results: list[ScoredPoint]) -> list[VectorSearchResult]:
        search_results = []

        for point in results:
            content = point.payload.get("content", "") if point.payload else ""

            metadata = {k: v for k, v in (point.payload or {}).items() if k != "content"}

            score = point.score

            search_results.append(VectorSearchResult(content=content, metadata=metadata, score=score))

        return search_results

    async def _delete_impl(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        try:
            if not self.initialized:
                await self.initialize()

            client = await self._get_async_client()

            if ids:
                # Convert ids to list that may include UUIDs
                point_ids: list[int | str | UUID] = list(ids)
                await client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=point_ids),
                )
            elif where:
                filter_condition = self._create_filter(where)
                if filter_condition is None:
                    return False
                await client.delete(
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(filter=filter_condition),
                )
            else:
                await client.delete(
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(filter=Filter()),
                )

            return True

        except Exception as e:
            logger.error(f"Error deleting documents from Qdrant: {e}", exc_info=True)
            return False

    async def _update_impl(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        try:
            if not self.initialized:
                await self.initialize()

            client = await self._get_async_client()

            # Prepare points for update
            points = []
            for i, doc_id in enumerate(ids):
                # Get existing point
                existing_points = await client.retrieve(
                    collection_name=self.collection_name,
                    ids=[doc_id],
                    with_payload=True,
                    with_vectors=True,
                )

                if not existing_points:
                    continue

                existing_point = existing_points[0]
                payload = existing_point.payload or {}

                if metadatas and i < len(metadatas):
                    flattened_metadata = self._flatten_metadata(metadatas[i])
                    payload.update(flattened_metadata)

                if documents and i < len(documents):
                    payload["content"] = documents[i]

                vector: list[float] | list[list[float]] | dict[str, Any] | None = existing_point.vector
                if embeddings and i < len(embeddings):
                    vector = embeddings[i]
                elif documents and i < len(documents) and self.embedder:
                    vector = await self.embedder.embed(documents[i])

                if vector is None:
                    raise ValueError(f"Vector cannot be None for document {doc_id}")

                point = PointStruct(id=doc_id, vector=vector, payload=payload)
                points.append(point)

            if points:
                await client.upsert(collection_name=self.collection_name, points=points)

            return True

        except Exception as e:
            logger.error(f"Error updating documents in Qdrant: {e}", exc_info=True)
            return False

    async def _upsert_impl(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        try:
            if not self.initialized:
                await self.initialize()

            client = await self._get_async_client()

            points = []
            for i, doc_id in enumerate(ids):
                try:
                    import uuid

                    if isinstance(doc_id, str) and len(doc_id) == 36 and doc_id.count("-") == 4:
                        uuid.UUID(doc_id)
                        qdrant_id = doc_id
                    else:
                        qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
                except (ValueError, TypeError):
                    import uuid

                    qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))

                payload = {}

                if metadatas and i < len(metadatas):
                    flattened_metadata = self._flatten_metadata(metadatas[i])
                    payload.update(flattened_metadata)

                if documents and i < len(documents):
                    payload["content"] = documents[i]

                payload["original_id"] = doc_id

                vector: list[float] | None = None
                if embeddings and i < len(embeddings):
                    vector = embeddings[i]
                elif documents and i < len(documents) and self.embedder:
                    vector = await self.embedder.embed(documents[i])
                elif documents and i < len(documents):
                    if self.vector_size is None:
                        raise ValueError("vector_size must be specified when no embedder is provided")
                    vector = [0.0] * self.vector_size

                if vector is None:
                    raise ValueError("Vector cannot be None when creating point")

                point = PointStruct(id=qdrant_id, vector=vector, payload=payload)
                points.append(point)

            await client.upsert(collection_name=self.collection_name, points=points)

            return True

        except Exception as e:
            logger.error(f"Error upserting documents in Qdrant: {e}", exc_info=True)
            return False

    def _initialize_db_sync(self) -> bool:
        try:
            client = self._get_client()

            vector_size = self.vector_size
            if not vector_size and self.embedder:
                if hasattr(self.embedder, "dimensions"):
                    vector_size = self.embedder.dimensions
                elif hasattr(self.embedder, "model_dimensions"):
                    vector_size = self.embedder.model_dimensions
                elif hasattr(self.embedder, "get_embedding_dimension"):
                    vector_size = self.embedder.get_embedding_dimension()
                else:
                    try:
                        if hasattr(self.embedder, "initialize_sync"):
                            self.embedder.initialize_sync()
                        if hasattr(self.embedder, "embed_sync"):
                            test_embedding = self.embedder.embed_sync("test")
                            if test_embedding and len(test_embedding) > 0:
                                vector_size = len(test_embedding)
                                logger.info(f"Detected embedder dimension: {vector_size}")
                    except Exception as e:
                        logger.warning(f"Could not detect embedder dimension: {e}")
                    if not vector_size:
                        vector_size = 1536
                        logger.warning(f"Using default vector size: {vector_size}")

            if not vector_size:
                raise ValueError("Vector size must be specified either directly or through embedder")

            try:
                existing_collection = client.get_collection(self.collection_name)
                vectors_config = existing_collection.config.params.vectors
                if vectors_config is None:
                    raise ValueError("Collection has no vector configuration")
                if isinstance(vectors_config, dict):
                    if not vectors_config:
                        raise ValueError("Collection has empty vector configuration")
                    existing_dim = next(iter(vectors_config.values())).size
                else:
                    existing_dim = vectors_config.size
                if existing_dim != vector_size:
                    logger.warning(
                        f"Collection '{self.collection_name}' exists with dimension {existing_dim}, "
                        f"but embedder requires {vector_size}. Deleting and recreating collection."
                    )
                    client.delete_collection(self.collection_name)
                    client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=self._get_distance_metric()),
                    )
            except Exception:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=self._get_distance_metric()),
                )

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}", exc_info=True)
            return False

    def _add_documents_impl_sync(self, documents: list[VectorDocument]) -> bool:
        try:
            if not self.exists():
                self._initialize_db_sync()

            client = self._get_client()

            points = []
            for document in documents:
                cleaned_content = document.content.replace("\x00", "\ufffd")

                if not document.id:
                    doc_id = hashlib.md5(cleaned_content.encode()).hexdigest()
                else:
                    doc_id = document.id

                try:
                    existing_points = client.retrieve(
                        collection_name=self.collection_name,
                        ids=[doc_id],
                        with_payload=False,
                        with_vectors=False,
                    )
                    if existing_points:
                        continue
                except Exception:
                    pass

                metadata = document.metadata or {}
                flattened_metadata = self._flatten_metadata(metadata)

                payload = {"content": cleaned_content, **flattened_metadata}

                if self.embedder and document.vector:
                    vector = document.vector
                elif self.embedder:
                    vector = self.embedder.embed_sync(cleaned_content)
                elif document.vector:
                    vector = document.vector
                else:
                    if self.vector_size is None:
                        raise ValueError("vector_size must be specified when no embedder is provided")
                    vector = [0.0] * self.vector_size

                if vector is None:
                    raise ValueError("Vector cannot be None for PointStruct")
                point = PointStruct(id=doc_id, vector=vector, payload=payload)
                points.append(point)

            if points:
                client.upsert(collection_name=self.collection_name, points=points)

            return True

        except Exception as e:
            logger.error(f"Error adding documents to Qdrant: {e}", exc_info=True)
            return False

    def _search_impl_sync(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        try:
            if not self.exists():
                self._initialize_db_sync()

            client = self._get_client()

            query_response = client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )

            return self._format_search_results(query_response.points)

        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}", exc_info=True)
            return []

    def _search_text_impl_sync(self, query: str, top_k: int) -> list[VectorSearchResult]:
        try:
            if not self.exists():
                self._initialize_db_sync()

            if not self.embedder:
                raise ValueError("Embedder required for text search")

            query_vector = self.embedder.embed_sync(query)
            return self._search_impl_sync(query_vector, top_k)

        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}", exc_info=True)
            return []

    def _delete_impl_sync(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        try:
            if not self.initialized:
                self._initialize_db_sync()

            client = self._get_client()

            if ids:
                # Convert ids to list that may include UUIDs
                point_ids: list[int | str | UUID] = list(ids)
                client.delete(
                    collection_name=self.collection_name,
                    points_selector=PointIdsList(points=point_ids),
                )
            elif where:
                filter_condition = self._create_filter(where)
                if filter_condition is None:
                    return False
                client.delete(
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(filter=filter_condition),
                )
            else:
                client.delete(
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(filter=Filter()),
                )

            return True

        except Exception as e:
            logger.error(f"Error deleting documents from Qdrant: {e}", exc_info=True)
            return False

    def _update_impl_sync(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        try:
            if not self.initialized:
                self._initialize_db_sync()

            client = self._get_client()

            points = []
            for i, doc_id in enumerate(ids):
                existing_points = client.retrieve(
                    collection_name=self.collection_name,
                    ids=[doc_id],
                    with_payload=True,
                    with_vectors=True,
                )

                if not existing_points:
                    continue

                existing_point = existing_points[0]
                payload = existing_point.payload or {}

                if metadatas and i < len(metadatas):
                    flattened_metadata = self._flatten_metadata(metadatas[i])
                    payload.update(flattened_metadata)

                if documents and i < len(documents):
                    payload["content"] = documents[i]

                vector: list[float] | list[list[float]] | dict[str, Any] | None = existing_point.vector
                if embeddings and i < len(embeddings):
                    vector = embeddings[i]
                elif documents and i < len(documents) and self.embedder:
                    vector = self.embedder.embed_sync(documents[i])

                if vector is None:
                    raise ValueError(f"Vector cannot be None for document {doc_id}")

                point = PointStruct(id=doc_id, vector=vector, payload=payload)
                points.append(point)

            if points:
                client.upsert(collection_name=self.collection_name, points=points)

            return True

        except Exception as e:
            logger.error(f"Error updating documents in Qdrant: {e}", exc_info=True)
            return False

    def _upsert_impl_sync(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        try:
            if not self.initialized:
                self._initialize_db_sync()

            client = self._get_client()

            points = []
            for i, doc_id in enumerate(ids):
                try:
                    import uuid

                    if isinstance(doc_id, str) and len(doc_id) == 36 and doc_id.count("-") == 4:
                        uuid.UUID(doc_id)
                        qdrant_id = doc_id
                    else:
                        qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
                except (ValueError, TypeError):
                    import uuid

                    qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(doc_id)))

                payload = {}

                if metadatas and i < len(metadatas):
                    flattened_metadata = self._flatten_metadata(metadatas[i])
                    payload.update(flattened_metadata)

                if documents and i < len(documents):
                    payload["content"] = documents[i]

                payload["original_id"] = doc_id

                vector: list[float] | None = None
                if embeddings and i < len(embeddings):
                    vector = embeddings[i]
                elif documents and i < len(documents) and self.embedder:
                    vector = self.embedder.embed_sync(documents[i])
                elif documents and i < len(documents):
                    if self.vector_size is None:
                        raise ValueError("vector_size must be specified when no embedder is provided")
                    vector = [0.0] * self.vector_size

                if vector is None:
                    raise ValueError("Vector cannot be None when creating point")

                point = PointStruct(id=qdrant_id, vector=vector, payload=payload)
                points.append(point)

            client.upsert(collection_name=self.collection_name, points=points)

            return True

        except Exception as e:
            logger.error(f"Error upserting documents in Qdrant: {e}", exc_info=True)
            return False
