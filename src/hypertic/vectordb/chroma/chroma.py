import hashlib
import json
from dataclasses import dataclass, field
from os import getenv
from typing import Any, Literal, cast

from hypertic.utils.log import get_logger
from hypertic.vectordb.base import BaseVectorDB, VectorDocument, VectorSearchResult

logger = get_logger(__name__)

try:
    import chromadb
    from chromadb import AsyncHttpClient, Client, HttpClient, PersistentClient
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import QueryResult
    from chromadb.config import Settings

    ClientAPI = Any
except ImportError as err:
    raise ImportError("`chromadb` not installed. Install with: pip install chromadb") from err


@dataclass
class ChromaDB(BaseVectorDB):
    collection: str
    embedder: Any | None = None
    path: str | None = None
    host: str | None = None
    port: int = 8000
    ssl: bool = False
    async_mode: bool = False
    chroma_cloud_api_key: str | None = None
    tenant: str | None = None
    database: str | None = None

    collection_name: str = field(init=False)

    _client: ClientAPI | None = field(default=None, init=False)
    _collection: Collection | None = field(default=None, init=False)

    def __post_init__(self):
        BaseVectorDB.__init__(self, embedder=self.embedder)
        self.collection_name = self.collection

    def _get_cloud_config(self) -> tuple[str | None, str | None, str | None]:
        api_key = self.chroma_cloud_api_key or getenv("CHROMA_API_KEY")
        tenant = self.tenant or getenv("CHROMA_TENANT")
        database = self.database or getenv("CHROMA_DATABASE")
        return api_key, tenant, database

    async def _get_client(self) -> ClientAPI:
        if self._client is None:
            if chromadb is None:
                raise ImportError("ChromaDB not installed. Install with: pip install chromadb")

            api_key, tenant, database = self._get_cloud_config()
            if api_key or self.chroma_cloud_api_key or self.tenant or self.database:
                if api_key and tenant and database:
                    self._client = chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key)
                elif api_key:
                    self._client = chromadb.CloudClient(api_key=api_key)
                else:
                    self._client = chromadb.CloudClient()
            elif self.host:
                if self.async_mode:
                    async_client = await AsyncHttpClient(host=self.host, port=self.port, ssl=self.ssl)
                    self._client = async_client
                else:
                    self._client = HttpClient(host=self.host, port=self.port, ssl=self.ssl)
            elif self.path:
                settings = Settings(anonymized_telemetry=False)
                self._client = PersistentClient(path=self.path, settings=settings)
            else:
                self._client = Client(
                    settings=Settings(anonymized_telemetry=False),
                )
        return self._client

    @property
    def client(self) -> ClientAPI:
        if self._client is None:
            if chromadb is None:
                raise ImportError("ChromaDB not installed. Install with: pip install chromadb")

            api_key, tenant, database = self._get_cloud_config()
            if api_key or self.chroma_cloud_api_key or self.tenant or self.database:
                if api_key and tenant and database:
                    self._client = chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key)
                elif api_key:
                    self._client = chromadb.CloudClient(api_key=api_key)
                else:
                    self._client = chromadb.CloudClient()
            elif self.host:
                self._client = HttpClient(host=self.host, port=self.port, ssl=self.ssl)
            elif self.path:
                settings = Settings(anonymized_telemetry=False)
                self._client = PersistentClient(path=self.path, settings=settings)

            else:
                self._client = Client(
                    settings=Settings(anonymized_telemetry=False),
                )
        return self._client

    async def _call_and_await_if_needed(self, callable_obj: Any, *args: Any, **kwargs: Any) -> None:
        if callable(callable_obj):
            result = callable_obj(*args, **kwargs)
            if hasattr(result, "__await__"):
                await result

    def _flatten_metadata(self, metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
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

    async def _initialize_db(self) -> bool:
        try:
            if self.async_mode and self.host:
                client = await self._get_client()
            else:
                client = self._get_client_sync()

            if self.async_mode and self.host:
                try:
                    collection_result = client.get_collection(name=self.collection_name)
                    if hasattr(collection_result, "__await__"):
                        self._collection = await collection_result
                    else:
                        self._collection = collection_result
                except Exception:
                    collection_result = client.create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"},
                    )
                    if hasattr(collection_result, "__await__"):
                        self._collection = await collection_result
                    else:
                        self._collection = collection_result
            else:
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                )

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}", exc_info=True)
            return False

    def exists(self) -> bool:
        try:
            self.client.get_collection(name=self.collection_name)
            return True
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
            if self.async_mode and self.host:
                client = await self._get_client()
                collection_result = client.get_collection(name=self.collection_name)
                if hasattr(collection_result, "__await__"):
                    collection = await collection_result
                else:
                    collection = collection_result
                include_list: list[Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]] = (
                    cast(
                        list[Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]],
                        include,
                    )
                    if include
                    else ["metadatas", "documents", "embeddings"]
                )
                get_result = collection.get(
                    ids=ids,
                    where=where,
                    limit=limit,
                    include=include_list,
                )
                if hasattr(get_result, "__await__"):
                    result = await get_result
                else:
                    result = get_result
                return dict(result) if result else {"ids": [], "metadatas": [], "documents": [], "embeddings": []}
            else:
                if self._collection is None:
                    raise RuntimeError("Collection not initialized")
                include_list_sync: list[Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]] = (
                    cast(
                        list[Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]],
                        include,
                    )
                    if include
                    else ["metadatas", "documents", "embeddings"]
                )
                result = self._collection.get(
                    ids=ids,
                    where=where,
                    limit=limit,
                    include=include_list_sync,
                )
                return dict(result) if result else {"ids": [], "metadatas": [], "documents": [], "embeddings": []}
        except Exception as e:
            logger.error(f"Error getting documents: {e}", exc_info=True)
            return {"ids": [], "metadatas": [], "documents": [], "embeddings": []}

    async def _count_documents_impl(self) -> int:
        try:
            if self.async_mode and self.host:
                client = await self._get_client()
                collection_result = client.get_collection(name=self.collection_name)
                if hasattr(collection_result, "__await__"):
                    collection = await collection_result
                else:
                    collection = collection_result
                count_result = collection.count()
                if hasattr(count_result, "__await__"):
                    result = await count_result
                else:
                    result = count_result
                return int(result) if result is not None else 0
            else:
                if self._collection is None:
                    raise RuntimeError("Collection not initialized")
                return int(self._collection.count())
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
            if self._collection is None:
                raise RuntimeError("Collection not initialized")
            default_include: list[Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]] = [
                "metadatas",
                "documents",
                "embeddings",
            ]
            include_list: list[Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]] = (
                cast(list[Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]], include) if include else default_include
            )
            result = self._collection.get(
                ids=ids,
                where=where,
                limit=limit,
                include=include_list,
            )
            return dict(result) if result else {"ids": [], "metadatas": [], "documents": [], "embeddings": []}
        except Exception as e:
            logger.error(f"Error getting documents: {e}", exc_info=True)
            return {"ids": [], "metadatas": [], "documents": [], "embeddings": []}

    def _count_documents_impl_sync(self) -> int:
        try:
            if self._collection is None:
                raise RuntimeError("Collection not initialized")
            return int(self._collection.count())
        except Exception as e:
            logger.error(f"Error counting documents: {e}", exc_info=True)
            return 0

    async def _add_documents_impl(self, documents: list[VectorDocument]) -> bool:
        try:
            if not self.initialized:
                success = await self.initialize()
                if not success:
                    raise Exception("Failed to initialize ChromaDB")

            ids = []
            docs = []
            docs_embeddings = []
            docs_metadata = []

            for document in documents:
                if not document.id:
                    doc_id = hashlib.md5(document.content.encode()).hexdigest()
                else:
                    doc_id = document.id

                cleaned_content = document.content.replace("\x00", "\ufffd")

                metadata = document.metadata or {}
                flattened_metadata = self._flatten_metadata(metadata)

                if self.embedder and document.vector:
                    embedding = document.vector
                elif self.embedder:
                    embedding = await self.embedder.embed(cleaned_content)
                else:
                    embedding = None

                ids.append(doc_id)
                docs.append(cleaned_content)
                docs_metadata.append(flattened_metadata)
                if embedding:
                    docs_embeddings.append(embedding)

            if self._collection is None:
                raise RuntimeError("Collection not initialized")

            cast_metadatas: Any = docs_metadata
            cast_embeddings: Any = docs_embeddings if docs_embeddings else None

            if docs_embeddings and len(docs_embeddings) == len(docs):
                if self.async_mode and self.host:
                    await self._call_and_await_if_needed(
                        self._collection.add,
                        ids=ids,
                        documents=docs,
                        metadatas=cast_metadatas,
                        embeddings=cast_embeddings,
                    )
                else:
                    self._collection.add(
                        ids=ids,
                        documents=docs,
                        metadatas=cast_metadatas,
                        embeddings=cast_embeddings,
                    )
            else:
                if self.async_mode and self.host:
                    await self._call_and_await_if_needed(self._collection.add, ids=ids, documents=docs, metadatas=cast_metadatas)
                else:
                    self._collection.add(ids=ids, documents=docs, metadatas=cast_metadatas)

            return True

        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}", exc_info=True)
            return False

    async def _search_impl(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        try:
            if not self.initialized:
                success = await self.initialize()
                if not success:
                    raise Exception("Failed to initialize ChromaDB")

            if self._collection is None:
                raise RuntimeError("Collection not initialized")

            cast_query_embeddings: Any = [query_vector]

            if self.async_mode and self.host:
                query_result = self._collection.query(
                    query_embeddings=cast_query_embeddings,
                    n_results=top_k,
                    include=["metadatas", "documents", "distances"],
                )
                results = query_result
            else:
                results = self._collection.query(
                    query_embeddings=cast_query_embeddings,
                    n_results=top_k,
                    include=["metadatas", "documents", "distances"],
                )

            return self._format_search_results(results)

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}", exc_info=True)
            return []

    async def _search_text_impl(self, query: str, top_k: int) -> list[VectorSearchResult]:
        try:
            if not self.initialized:
                success = await self.initialize()
                if not success:
                    raise Exception("Failed to initialize ChromaDB")

            if self._collection is None:
                raise RuntimeError("Collection not initialized")

            if self.async_mode and self.host:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["metadatas", "documents", "distances"],
                )
            else:
                if self._collection is None:
                    raise RuntimeError("Collection not initialized")
                results = self._collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["metadatas", "documents", "distances"],
                )

            return self._format_search_results(results)

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}", exc_info=True)
            return []

    def _format_search_results(self, results: QueryResult) -> list[VectorSearchResult]:
        search_results: list[VectorSearchResult] = []

        if not results or not results.get("documents") or not results["documents"] or not results["documents"][0]:
            return search_results

        ids_list = results.get("ids", [[]])
        metadata_list = results.get("metadatas", [[{}]])
        documents_list = results.get("documents", [[]])
        distances_list = results.get("distances", [[]])

        if not ids_list or not metadata_list or not documents_list or not distances_list:
            return search_results

        ids = ids_list[0]
        metadata = [dict(m) if m else {} for m in metadata_list[0]]
        documents = documents_list[0]
        distances = distances_list[0]

        for _idx, (_doc_id, doc_metadata, document, distance) in enumerate(zip(ids, metadata, documents, distances, strict=False)):
            score = 1.0 - distance if distance <= 1.0 else 0.0

            search_results.append(VectorSearchResult(content=str(document) if document else "", metadata=doc_metadata, score=score))

        return search_results

    async def _delete_impl(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        try:
            if not self.initialized:
                await self.initialize()

            if self._collection is None:
                raise RuntimeError("Collection not initialized")

            if self.async_mode and self.host:
                if ids is not None:
                    await self._call_and_await_if_needed(self._collection.delete, ids=ids)
                elif where is not None:
                    await self._call_and_await_if_needed(self._collection.delete, where=where)
            else:
                if ids is not None:
                    self._collection.delete(ids=ids)
                elif where is not None:
                    self._collection.delete(where=where)
            return True

        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}", exc_info=True)
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

            if self._collection is None:
                raise RuntimeError("Collection not initialized")

            flattened_metadatas: list[dict[str, str | int | float | bool]] | None = None
            if metadatas is not None:
                flattened_metadatas = [self._flatten_metadata(meta) for meta in metadatas]

            generated_embeddings: list[list[float]] | None = None
            if documents is not None and self.embedder and embeddings is None:
                generated_embeddings = []
                for doc in documents:
                    embedding = await self.embedder.embed(doc)
                    generated_embeddings.append(embedding)

            final_embeddings = embeddings if embeddings is not None else generated_embeddings

            cast_metadatas: Any = flattened_metadatas
            cast_embeddings: Any = final_embeddings

            if self.async_mode and self.host:
                if cast_metadatas is not None and documents is not None and cast_embeddings is not None:
                    await self._call_and_await_if_needed(
                        self._collection.update,
                        ids=ids,
                        documents=documents,
                        metadatas=cast_metadatas,
                        embeddings=cast_embeddings,
                    )
                elif cast_metadatas is not None and documents is not None:
                    await self._call_and_await_if_needed(
                        self._collection.update,
                        ids=ids,
                        documents=documents,
                        metadatas=cast_metadatas,
                    )
                elif documents is not None and cast_embeddings is not None:
                    await self._call_and_await_if_needed(
                        self._collection.update,
                        ids=ids,
                        documents=documents,
                        embeddings=cast_embeddings,
                    )
                elif cast_metadatas is not None:
                    await self._call_and_await_if_needed(self._collection.update, ids=ids, metadatas=cast_metadatas)
                elif cast_embeddings is not None:
                    await self._call_and_await_if_needed(self._collection.update, ids=ids, embeddings=cast_embeddings)
                else:
                    await self._call_and_await_if_needed(self._collection.update, ids=ids)
            else:
                if cast_metadatas is not None and documents is not None and cast_embeddings is not None:
                    self._collection.update(
                        ids=ids,
                        documents=documents,
                        metadatas=cast_metadatas,
                        embeddings=cast_embeddings,
                    )
                elif cast_metadatas is not None and documents is not None:
                    self._collection.update(ids=ids, documents=documents, metadatas=cast_metadatas)
                elif documents is not None and cast_embeddings is not None:
                    self._collection.update(ids=ids, documents=documents, embeddings=cast_embeddings)
                elif cast_metadatas is not None:
                    self._collection.update(ids=ids, metadatas=cast_metadatas)
                elif cast_embeddings is not None:
                    self._collection.update(ids=ids, embeddings=cast_embeddings)
                else:
                    self._collection.update(ids=ids)
            return True

        except Exception as e:
            logger.error(f"Error updating documents in ChromaDB: {e}", exc_info=True)
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

            if self._collection is None:
                raise RuntimeError("Collection not initialized")

            flattened_metadatas: list[dict[str, str | int | float | bool]] | None = None
            if metadatas is not None:
                flattened_metadatas = [self._flatten_metadata(meta) for meta in metadatas]

            generated_embeddings: list[list[float]] | None = None
            if documents is not None and self.embedder and embeddings is None:
                generated_embeddings = []
                for doc in documents:
                    embedding = await self.embedder.embed(doc)
                    generated_embeddings.append(embedding)

            final_embeddings = embeddings if embeddings is not None else generated_embeddings

            cast_metadatas: Any = flattened_metadatas
            cast_embeddings: Any = final_embeddings

            if self.async_mode and self.host:
                if cast_metadatas is not None and documents is not None and cast_embeddings is not None:
                    await self._call_and_await_if_needed(
                        self._collection.upsert,
                        ids=ids,
                        documents=documents,
                        metadatas=cast_metadatas,
                        embeddings=cast_embeddings,
                    )
                elif cast_metadatas is not None and documents is not None:
                    await self._call_and_await_if_needed(
                        self._collection.upsert,
                        ids=ids,
                        documents=documents,
                        metadatas=cast_metadatas,
                    )
                elif documents is not None and cast_embeddings is not None:
                    await self._call_and_await_if_needed(
                        self._collection.upsert,
                        ids=ids,
                        documents=documents,
                        embeddings=cast_embeddings,
                    )
                elif cast_metadatas is not None:
                    await self._call_and_await_if_needed(self._collection.upsert, ids=ids, metadatas=cast_metadatas)
                elif cast_embeddings is not None:
                    await self._call_and_await_if_needed(self._collection.upsert, ids=ids, embeddings=cast_embeddings)
                else:
                    await self._call_and_await_if_needed(self._collection.upsert, ids=ids)
            else:
                if cast_metadatas is not None and documents is not None and cast_embeddings is not None:
                    self._collection.upsert(
                        ids=ids,
                        documents=documents,
                        metadatas=cast_metadatas,
                        embeddings=cast_embeddings,
                    )
                elif cast_metadatas is not None and documents is not None:
                    self._collection.upsert(ids=ids, documents=documents, metadatas=cast_metadatas)
                elif documents is not None and cast_embeddings is not None:
                    self._collection.upsert(ids=ids, documents=documents, embeddings=cast_embeddings)
                elif cast_metadatas is not None:
                    self._collection.upsert(ids=ids, metadatas=cast_metadatas)
                elif cast_embeddings is not None:
                    self._collection.upsert(ids=ids, embeddings=cast_embeddings)
                else:
                    self._collection.upsert(ids=ids)
            return True

        except Exception as e:
            logger.error(f"Error upserting documents in ChromaDB: {e}", exc_info=True)
            return False

    def _initialize_db_sync(self) -> bool:
        try:
            client = self._get_client_sync()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
            )
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}", exc_info=True)
            return False

    def _get_client_sync(self) -> ClientAPI:
        if self._client is None:
            if chromadb is None:
                raise ImportError("ChromaDB not installed. Install with: pip install chromadb")

            api_key, tenant, database = self._get_cloud_config()
            if api_key or self.chroma_cloud_api_key or self.tenant or self.database:
                if api_key and tenant and database:
                    self._client = chromadb.CloudClient(tenant=tenant, database=database, api_key=api_key)
                elif api_key:
                    self._client = chromadb.CloudClient(api_key=api_key)
                else:
                    self._client = chromadb.CloudClient()
            elif self.host:
                settings = Settings(
                    chroma_server_host=self.host,
                    chroma_server_http_port=self.port,
                    chroma_server_ssl_enabled=self.ssl,
                )
                self._client = chromadb.HttpClient(settings=settings)
            elif self.path:
                settings = Settings(anonymized_telemetry=False)
                self._client = PersistentClient(path=self.path, settings=settings)
            else:
                self._client = chromadb.Client()

        return self._client

    def _add_documents_impl_sync(self, documents: list[VectorDocument]) -> bool:
        try:
            if not self.initialized:
                self._initialize_db_sync()

            ids = [doc.id for doc in documents]
            documents_text = [doc.content for doc in documents]
            metadatas_list: list[dict[str, Any] | None] = [doc.metadata for doc in documents]
            embeddings_list: list[list[float] | None] = [doc.vector for doc in documents]

            flattened_metadatas: list[dict[str, str | int | float | bool]] = []
            for meta in metadatas_list:
                if meta is not None:
                    flattened_metadatas.append(self._flatten_metadata(meta))
                else:
                    flattened_metadatas.append({})

            filtered_embeddings: list[list[float]] = [emb for emb in embeddings_list if emb is not None]

            if self._collection is None:
                raise RuntimeError("Collection not initialized")

            cast_metadatas_sync: Any = flattened_metadatas
            cast_embeddings_sync: Any = filtered_embeddings if filtered_embeddings else None

            if cast_embeddings_sync is not None:
                self._collection.add(
                    ids=ids,
                    documents=documents_text,
                    metadatas=cast_metadatas_sync,
                    embeddings=cast_embeddings_sync,
                )
            else:
                self._collection.add(ids=ids, documents=documents_text, metadatas=cast_metadatas_sync)
            return True

        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}", exc_info=True)
            return False

    def _search_impl_sync(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        try:
            if not self.initialized:
                self._initialize_db_sync()

            if self._collection is None:
                raise RuntimeError("Collection not initialized")

            cast_query_embeddings_sync: Any = [query_vector]

            results = self._collection.query(
                query_embeddings=cast_query_embeddings_sync,
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            search_results: list[VectorSearchResult] = []
            documents_list = results.get("documents")
            metadatas_list = results.get("metadatas")
            distances_list = results.get("distances")

            if documents_list and documents_list[0] and metadatas_list and metadatas_list[0] and distances_list and distances_list[0]:
                for _i, (doc, metadata, distance) in enumerate(zip(documents_list[0], metadatas_list[0], distances_list[0], strict=False)):
                    score = 1 - distance
                    metadata_dict: dict[str, Any] = dict(metadata) if metadata else {}
                    search_results.append(VectorSearchResult(content=doc, metadata=metadata_dict, score=score))

            return search_results

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}", exc_info=True)
            return []

    def _search_text_impl_sync(self, query: str, top_k: int) -> list[VectorSearchResult]:
        try:
            if not self.initialized:
                self._initialize_db_sync()

            if self._collection is None:
                raise RuntimeError("Collection not initialized")
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            search_results: list[VectorSearchResult] = []
            documents_list = results.get("documents")
            metadatas_list = results.get("metadatas")
            distances_list = results.get("distances")

            if documents_list and documents_list[0] and metadatas_list and metadatas_list[0] and distances_list and distances_list[0]:
                for _i, (doc, metadata, distance) in enumerate(zip(documents_list[0], metadatas_list[0], distances_list[0], strict=False)):
                    score = 1 - distance
                    metadata_dict: dict[str, Any] = dict(metadata) if metadata else {}
                    search_results.append(VectorSearchResult(content=doc, metadata=metadata_dict, score=score))

            return search_results

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}", exc_info=True)
            return []

    def _delete_impl_sync(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        try:
            if not self.initialized:
                self._initialize_db_sync()

            if ids:
                if self._collection is None:
                    raise RuntimeError("Collection not initialized")
                self._collection.delete(ids=ids)
            elif where:
                if self._collection is None:
                    raise RuntimeError("Collection not initialized")
                self._collection.delete(where=where)
            else:
                if self._collection is None:
                    raise RuntimeError("Collection not initialized")
                self._collection.delete()

            return True

        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}", exc_info=True)
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

            if self._collection is None:
                raise RuntimeError("Collection not initialized")

            flattened_metadatas: list[dict[str, str | int | float | bool]] | None = None
            if metadatas is not None:
                flattened_metadatas = [self._flatten_metadata(meta) for meta in metadatas]

            generated_embeddings: list[list[float]] | None = None
            if documents is not None and self.embedder and embeddings is None:
                generated_embeddings = []
                for doc in documents:
                    embedding = self.embedder.embed_sync(doc)
                    generated_embeddings.append(embedding)

            final_embeddings = embeddings if embeddings is not None else generated_embeddings

            cast_metadatas_sync: Any = flattened_metadatas
            cast_embeddings_sync: Any = final_embeddings

            if cast_metadatas_sync is not None and documents is not None and cast_embeddings_sync is not None:
                self._collection.update(
                    ids=ids,
                    documents=documents,
                    metadatas=cast_metadatas_sync,
                    embeddings=cast_embeddings_sync,
                )
            elif cast_metadatas_sync is not None and documents is not None:
                self._collection.update(ids=ids, documents=documents, metadatas=cast_metadatas_sync)
            elif documents is not None and cast_embeddings_sync is not None:
                self._collection.update(ids=ids, documents=documents, embeddings=cast_embeddings_sync)
            elif cast_metadatas_sync is not None:
                self._collection.update(ids=ids, metadatas=cast_metadatas_sync)
            elif cast_embeddings_sync is not None:
                self._collection.update(ids=ids, embeddings=cast_embeddings_sync)
            else:
                self._collection.update(ids=ids)
            return True

        except Exception as e:
            logger.error(f"Error updating documents in ChromaDB: {e}", exc_info=True)
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

            if self._collection is None:
                raise RuntimeError("Collection not initialized")

            flattened_metadatas: list[dict[str, str | int | float | bool]] | None = None
            if metadatas is not None:
                flattened_metadatas = [self._flatten_metadata(meta) for meta in metadatas]

            generated_embeddings: list[list[float]] | None = None
            if documents is not None and embeddings is None and self.embedder:
                generated_embeddings = [self.embedder.embed_sync(doc) for doc in documents]

            final_embeddings = embeddings if embeddings is not None else generated_embeddings

            cast_metadatas_sync: Any = flattened_metadatas
            cast_embeddings_sync: Any = final_embeddings

            if cast_metadatas_sync is not None and documents is not None and cast_embeddings_sync is not None:
                self._collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=cast_metadatas_sync,
                    embeddings=cast_embeddings_sync,
                )
            elif cast_metadatas_sync is not None and documents is not None:
                self._collection.upsert(ids=ids, documents=documents, metadatas=cast_metadatas_sync)
            elif documents is not None and cast_embeddings_sync is not None:
                self._collection.upsert(ids=ids, documents=documents, embeddings=cast_embeddings_sync)
            elif cast_metadatas_sync is not None:
                self._collection.upsert(ids=ids, metadatas=cast_metadatas_sync)
            elif cast_embeddings_sync is not None:
                self._collection.upsert(ids=ids, embeddings=cast_embeddings_sync)
            else:
                self._collection.upsert(ids=ids)
            return True

        except Exception as e:
            logger.error(f"Error upserting documents in ChromaDB: {e}", exc_info=True)
            return False
