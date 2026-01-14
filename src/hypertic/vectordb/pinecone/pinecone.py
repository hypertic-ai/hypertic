from dataclasses import dataclass, field
from os import getenv
from typing import Any

from hypertic.utils.log import get_logger
from hypertic.vectordb.base import BaseVectorDB, VectorDocument, VectorSearchResult

logger = get_logger(__name__)

try:
    from pinecone import AwsRegion, CloudProvider, Pinecone, ServerlessSpec, VectorType
    from pinecone.db_data import _Index as Index
except ImportError as err:
    raise ImportError("`pinecone` not installed. Install with: pip install pinecone") from err


@dataclass
class PineconeDB(BaseVectorDB):
    collection: str
    embedder: Any | None = None
    api_key: str | None = None
    environment: str | None = None
    dimension: int | None = None
    metric: str = "cosine"
    cloud_provider: str = "aws"
    region: str = "us-east-1"
    vector_type: str = "dense"

    collection_name: str = field(init=False)

    _client: Pinecone | None = field(default=None, init=False)
    _index: Index | None = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self):
        BaseVectorDB.__init__(self, embedder=self.embedder)

        self.collection_name = self.collection

    def _get_api_key(self) -> str:
        api_key = self.api_key or getenv("PINECONE_API_KEY")
        if api_key is None:
            raise ValueError("Pinecone API key is required. Pass api_key parameter or set PINECONE_API_KEY environment variable.")
        return api_key

    def _get_cloud_provider(self) -> CloudProvider:
        provider_map = {
            "aws": CloudProvider.AWS,
            "gcp": CloudProvider.GCP,
            "azure": CloudProvider.AZURE,
        }
        return provider_map.get(self.cloud_provider.lower(), CloudProvider.AWS)

    def _get_region(self) -> AwsRegion:
        region_map = {
            "us-east-1": AwsRegion.US_EAST_1,
            "us-west-2": AwsRegion.US_WEST_2,
            "eu-west-1": AwsRegion.EU_WEST_1,
        }
        return region_map.get(self.region.lower(), AwsRegion.US_EAST_1)

    def _get_vector_type(self) -> VectorType:
        type_map = {"dense": VectorType.DENSE, "sparse": VectorType.SPARSE}
        return type_map.get(self.vector_type.lower(), VectorType.DENSE)

    def _flatten_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        import json

        if not isinstance(metadata, dict):
            if metadata is None:
                return {}
            elif isinstance(metadata, str | int | float | bool):
                return {"value": metadata}
            else:
                return {"value": json.dumps(metadata)}

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
            else:
                flattened[prefix] = obj

        for key, value in metadata.items():
            _flatten_recursive(value, key)

        return flattened

    def _get_client(self) -> Pinecone:
        if self._client is None:
            if Pinecone is None:
                raise ImportError("Pinecone not installed. Install with: pip install pinecone")

            api_key = self._get_api_key()
            if not api_key:
                raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY environment variable or pass api_key parameter.")

            if api_key is not None:
                self._client = Pinecone(api_key=api_key)
            else:
                self._client = Pinecone()

        return self._client

    def _get_index(self) -> Index:
        if self._index is None:
            client = self._get_client()
            self._index = client.Index(self.collection_name)

        return self._index

    async def _initialize_db(self) -> bool:
        try:
            if self._initialized:
                return True

            client = self._get_client()

            existing_indexes = client.list_indexes()
            index_names = [idx.name for idx in existing_indexes]

            if self.collection_name not in index_names:
                if self.dimension is None:
                    if self.embedder and hasattr(self.embedder, "dimensions"):
                        self.dimension = self.embedder.dimensions
                    else:
                        raise ValueError("Dimension must be specified when creating a new index")

                spec = ServerlessSpec(cloud=self._get_cloud_provider(), region=self._get_region())
                vector_type = self._get_vector_type()

                client.create_index(
                    name=self.collection_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=spec,
                    vector_type=vector_type,
                )
                logger.info(f"Created Pinecone index: {self.collection_name}")

            self._get_index()
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing Pinecone database: {e}", exc_info=True)
            return False

    def _initialize_db_sync(self) -> bool:
        try:
            if self._initialized:
                return True

            if self.embedder:
                self.embedder.initialize_sync()

            client = self._get_client()

            if not client.has_index(self.collection_name):
                spec = ServerlessSpec(cloud=self._get_cloud_provider(), region=self._get_region())

                client.create_index(
                    name=self.collection_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=spec,
                )

                import time

                while not client.describe_index(self.collection_name).status["ready"]:
                    time.sleep(1)

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing Pinecone database: {e}", exc_info=True)
            return False

    async def _add_documents_impl(self, documents: list[VectorDocument]) -> bool:
        try:
            if not self._initialized:
                await self._initialize_db()

            index = self._get_index()

            vectors = []
            for doc in documents:
                if doc.vector is None:
                    logger.warning(f"Document {doc.id} has no vector, skipping")
                    continue

                metadata = doc.metadata or {}
                if doc.content:
                    metadata["content"] = doc.content

                flattened_metadata = self._flatten_metadata(metadata)

                vectors.append((doc.id, doc.vector, flattened_metadata))

            if not vectors:
                logger.warning("No valid vectors to upsert")
                return False

            index.upsert(vectors=vectors)
            return True

        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {e}", exc_info=True)
            return False

    def _add_documents_impl_sync(self, documents: list[VectorDocument]) -> bool:
        try:
            if not self._initialized:
                self._initialize_db_sync()

            index = self._get_index()

            vectors = []
            for doc in documents:
                if doc.vector is None:
                    logger.warning(f"Document {doc.id} has no vector, skipping")
                    continue

                metadata = doc.metadata or {}
                if doc.content:
                    metadata["content"] = doc.content

                flattened_metadata = self._flatten_metadata(metadata)

                vectors.append((doc.id, doc.vector, flattened_metadata))

            if not vectors:
                logger.warning("No valid vectors to upsert")
                return False

            index.upsert(vectors=vectors)
            return True

        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {e}", exc_info=True)
            return False

    async def _search_impl(self, query_vector: list[float], top_k: int = 5, filters: dict[str, Any] | None = None) -> list[VectorSearchResult]:
        try:
            if not self._initialized:
                await self._initialize_db()

            index = self._get_index()

            query_params: dict[str, Any] = {"vector": query_vector, "top_k": top_k, "include_metadata": True}

            if filters:
                query_params["filter"] = filters

            response = index.query(**query_params)

            # Handle both QueryResponse and ApplyResult types
            if hasattr(response, "matches"):
                matches = response.matches
            else:
                # If it's an ApplyResult, we can't access matches directly
                logger.warning("Unexpected response type from Pinecone query")
                return []

            results = []
            for match in matches:
                results.append(
                    VectorSearchResult(
                        content=match.metadata.get("content", ""),
                        metadata=match.metadata,
                        score=float(match.score),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}", exc_info=True)
            return []

    def _search_impl_sync(self, query_vector: list[float], top_k: int = 5, filters: dict[str, Any] | None = None) -> list[VectorSearchResult]:
        try:
            if not self._initialized:
                self._initialize_db_sync()

            index = self._get_index()

            query_params: dict[str, Any] = {"vector": query_vector, "top_k": top_k, "include_metadata": True}

            if filters:
                query_params["filter"] = filters

            response = index.query(**query_params)

            # Handle both QueryResponse and ApplyResult types
            if hasattr(response, "matches"):
                matches = response.matches
            else:
                # If it's an ApplyResult, we can't access matches directly
                logger.warning("Unexpected response type from Pinecone query")
                return []

            results = []
            for match in matches:
                results.append(
                    VectorSearchResult(
                        content=match.metadata.get("content", ""),
                        metadata=match.metadata,
                        score=float(match.score),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}", exc_info=True)
            return []

    async def _search_text_impl(self, query: str, top_k: int) -> list[VectorSearchResult]:
        if self.embedder:
            query_vector = await self.embedder.embed(query)
            return await self._search_impl(query_vector, top_k)
        else:
            logger.warning("Text search requires an embedder")
            return []

    def _search_text_impl_sync(self, query: str, top_k: int) -> list[VectorSearchResult]:
        if self.embedder:
            query_vector = self.embedder.embed_sync(query)
            return self._search_impl_sync(query_vector, top_k)
        else:
            logger.warning("Text search requires an embedder")
            return []

    async def _delete_impl(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        try:
            if not self._initialized:
                await self._initialize_db()

            index = self._get_index()

            if ids:
                index.delete(ids=ids)
            elif where:
                index.delete(filter=where)
            else:
                logger.warning("No IDs or filter provided for deletion")
                return False

            return True

        except Exception as e:
            logger.error(f"Error deleting documents from Pinecone: {e}", exc_info=True)
            return False

    def _delete_impl_sync(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        try:
            if not self._initialized:
                self._initialize_db_sync()

            index = self._get_index()

            if ids:
                index.delete(ids=ids)
            elif where:
                index.delete(filter=where)
            else:
                logger.warning("No IDs or filter provided for deletion")
                return False

            return True

        except Exception as e:
            logger.error(f"Error deleting documents from Pinecone: {e}", exc_info=True)
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

            index = self._get_index()

            if metadatas and not documents and not embeddings:
                update_vectors = []
                for i, doc_id in enumerate(ids):
                    if i < len(metadatas):
                        flattened_metadata = self._flatten_metadata(metadatas[i])
                        update_vectors.append({"id": doc_id, "metadata": flattened_metadata})

                if update_vectors:
                    for update_vector in update_vectors:
                        index.update(id=str(update_vector["id"]), metadata=update_vector["metadata"])
            else:
                vectors = []
                for i, doc_id in enumerate(ids):
                    if embeddings and i < len(embeddings):
                        vector = embeddings[i]
                    else:
                        fetch_response = index.fetch(ids=[doc_id])
                        if doc_id in fetch_response.vectors:
                            vector = fetch_response.vectors[doc_id].values
                        else:
                            logger.warning(f"Document {doc_id} not found for update")
                            continue

                    metadata = {}
                    if metadatas and i < len(metadatas):
                        flattened_metadata = self._flatten_metadata(metadatas[i])
                        metadata.update(flattened_metadata)

                    if documents and i < len(documents):
                        metadata["content"] = documents[i]

                    vectors.append((doc_id, vector, metadata))

                if vectors:
                    index.upsert(vectors=vectors)

            return True

        except Exception as e:
            logger.error(f"Error updating documents in Pinecone: {e}", exc_info=True)
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

            index = self._get_index()

            if metadatas and not documents and not embeddings:
                update_vectors = []
                for i, doc_id in enumerate(ids):
                    if i < len(metadatas):
                        flattened_metadata = self._flatten_metadata(metadatas[i])
                        update_vectors.append({"id": doc_id, "metadata": flattened_metadata})

                if update_vectors:
                    for update_vector in update_vectors:
                        index.update(id=str(update_vector["id"]), metadata=update_vector["metadata"])
            else:
                vectors = []
                for i, doc_id in enumerate(ids):
                    if embeddings and i < len(embeddings):
                        vector = embeddings[i]
                    else:
                        fetch_response = index.fetch(ids=[doc_id])
                        if doc_id in fetch_response.vectors:
                            vector = fetch_response.vectors[doc_id].values
                        else:
                            logger.warning(f"Document {doc_id} not found for update")
                            continue

                    metadata = {}
                    if metadatas and i < len(metadatas):
                        flattened_metadata = self._flatten_metadata(metadatas[i])
                        metadata.update(flattened_metadata)

                    if documents and i < len(documents):
                        metadata["content"] = documents[i]

                    vectors.append((doc_id, vector, metadata))

                if vectors:
                    index.upsert(vectors=vectors)

            return True

        except Exception as e:
            logger.error(f"Error updating documents in Pinecone: {e}", exc_info=True)
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

            index = self._get_index()

            vectors = []
            for i, doc_id in enumerate(ids):
                if embeddings and i < len(embeddings):
                    vector = embeddings[i]
                elif self.embedder and documents and i < len(documents):
                    vector = await self.embedder.embed(documents[i])
                else:
                    logger.warning(f"No embedding available for document {doc_id}")
                    continue

                metadata = {}
                if metadatas and i < len(metadatas):
                    flattened_metadata = self._flatten_metadata(metadatas[i])
                    metadata.update(flattened_metadata)

                if documents and i < len(documents):
                    metadata["content"] = documents[i]

                vectors.append((doc_id, vector, metadata))

            if vectors:
                index.upsert(vectors=vectors)

            return True

        except Exception as e:
            logger.error(f"Error upserting documents in Pinecone: {e}", exc_info=True)
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

            index = self._get_index()

            vectors = []
            for i, doc_id in enumerate(ids):
                if embeddings and i < len(embeddings):
                    vector = embeddings[i]
                elif self.embedder and documents and i < len(documents):
                    vector = self.embedder.embed_sync(documents[i])
                else:
                    logger.warning(f"No embedding available for document {doc_id}")
                    continue

                metadata = {}
                if metadatas and i < len(metadatas):
                    flattened_metadata = self._flatten_metadata(metadatas[i])
                    metadata.update(flattened_metadata)

                if documents and i < len(documents):
                    metadata["content"] = documents[i]

                vectors.append((doc_id, vector, metadata))

            if vectors:
                index.upsert(vectors=vectors)

            return True

        except Exception as e:
            logger.error(f"Error upserting documents in Pinecone: {e}", exc_info=True)
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

            index = self._get_index()

            if ids:
                fetch_response = index.fetch(ids=ids)
                # Handle both FetchResponse and ApplyResult types
                if hasattr(fetch_response, "vectors"):
                    vectors_dict = fetch_response.vectors
                else:
                    logger.warning("Unexpected response type from Pinecone fetch")
                    return {"ids": [], "contents": [], "metadatas": []}

                result_ids = []
                contents = []
                metadatas = []

                for doc_id in ids:
                    if doc_id in vectors_dict:
                        vector_data = vectors_dict[doc_id]
                        result_ids.append(doc_id)
                        metadata = vector_data.metadata if vector_data.metadata else {}
                        contents.append(metadata.get("content", ""))
                        metadatas.append(metadata)

                return {"ids": result_ids, "contents": contents, "metadatas": metadatas}
            else:
                query_vector = [0.0] * self.dimension if self.dimension else [0.0] * 1536
                query_response = index.query(vector=query_vector, top_k=limit or 1000, include_metadata=True)

                # Handle both QueryResponse and ApplyResult types
                if hasattr(query_response, "matches"):
                    matches = query_response.matches
                else:
                    logger.warning("Unexpected response type from Pinecone query")
                    return {"ids": [], "contents": [], "metadatas": []}

                result_ids = []
                contents = []
                metadatas = []

                for match in matches:
                    result_ids.append(match.id)
                    metadata = match.metadata if match.metadata else {}
                    contents.append(metadata.get("content", ""))
                    metadatas.append(metadata)

                return {"ids": result_ids, "contents": contents, "metadatas": metadatas}

        except Exception as e:
            logger.error(f"Error getting documents from Pinecone: {e}", exc_info=True)
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

            index = self._get_index()

            if ids:
                fetch_response = index.fetch(ids=ids)
                # Handle both FetchResponse and ApplyResult types
                if hasattr(fetch_response, "vectors"):
                    vectors_dict = fetch_response.vectors
                else:
                    logger.warning("Unexpected response type from Pinecone fetch")
                    return {"ids": [], "contents": [], "metadatas": []}

                result_ids = []
                contents = []
                metadatas = []

                for doc_id in ids:
                    if doc_id in vectors_dict:
                        vector_data = vectors_dict[doc_id]
                        result_ids.append(doc_id)
                        metadata = vector_data.metadata if vector_data.metadata else {}
                        contents.append(metadata.get("content", ""))
                        metadatas.append(metadata)

                return {"ids": result_ids, "contents": contents, "metadatas": metadatas}
            else:
                query_vector = [0.0] * self.dimension if self.dimension else [0.0] * 1536
                query_response = index.query(vector=query_vector, top_k=limit or 1000, include_metadata=True)

                # Handle both QueryResponse and ApplyResult types
                if hasattr(query_response, "matches"):
                    matches = query_response.matches
                else:
                    logger.warning("Unexpected response type from Pinecone query")
                    return {"ids": [], "contents": [], "metadatas": []}

                result_ids = []
                contents = []
                metadatas = []

                for match in matches:
                    result_ids.append(match.id)
                    metadata = match.metadata if match.metadata else {}
                    contents.append(metadata.get("content", ""))
                    metadatas.append(metadata)

                return {"ids": result_ids, "contents": contents, "metadatas": metadatas}

        except Exception as e:
            logger.error(f"Error getting documents from Pinecone: {e}", exc_info=True)
            return {"ids": [], "contents": [], "metadatas": []}

    async def _count_documents_impl(self) -> int:
        try:
            if not self._initialized:
                await self._initialize_db()

            index = self._get_index()
            stats = index.describe_index_stats()
            count = stats.total_vector_count
            return int(count) if count is not None else 0

        except Exception as e:
            logger.error(f"Error counting documents in Pinecone: {e}", exc_info=True)
            return 0

    def _count_documents_impl_sync(self) -> int:
        try:
            if not self._initialized:
                self._initialize_db_sync()

            index = self._get_index()
            stats = index.describe_index_stats()
            count = stats.total_vector_count
            return int(count) if count is not None else 0

        except Exception as e:
            logger.error(f"Error counting documents in Pinecone: {e}", exc_info=True)
            return 0

    async def async_exists(self) -> bool:
        try:
            client = self._get_client()
            existing_indexes = client.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            return self.collection_name in index_names
        except Exception:
            return False

    def exists(self) -> bool:
        try:
            client = self._get_client()
            existing_indexes = client.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            return self.collection_name in index_names
        except Exception:
            return False
