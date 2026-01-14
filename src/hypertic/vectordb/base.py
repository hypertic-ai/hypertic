import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from hypertic.utils.log import get_logger
from hypertic.vectordb.document_loader import DocumentLoader

logger = get_logger(__name__)


@dataclass
class VectorDocument:
    """Document to be stored in vector database"""

    id: str
    content: str
    metadata: dict[str, Any] | None = None
    vector: list[float] | None = None


@dataclass
class VectorSearchResult:
    """Result from vector database search"""

    content: str
    metadata: dict[str, Any]
    score: float


class BaseVectorDB(ABC):
    """Base class for all vector databases"""

    def __init__(self, embedder=None, **kwargs):
        self.embedder = embedder
        self.initialized = False
        self._document_loader = DocumentLoader()

    async def initialize(self) -> bool:
        """Initialize the vector database and embedder"""
        if self.embedder:
            await self.embedder.initialize()
        return await self._initialize_db()

    @abstractmethod
    async def _initialize_db(self) -> bool:
        """Initialize the specific vector database"""
        pass

    async def async_add(
        self,
        files: list[str] | None = None,
        texts: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        chunking_strategy: str = "document",
        chunking_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> bool:
        """Add files or text to the vector database with automatic processing

        Args:
            files: List of file paths or URLs to load
            texts: List of text strings to add directly (alternative to files)
            metadatas: Optional list of metadata dicts (one per file/text)
            ids: Optional list of document IDs (one per file/text)
            chunking_strategy: Chunking strategy to use (only for files)
            chunking_params: Optional parameters for chunking
            **kwargs: Additional parameters

        Note: Either files or texts must be provided, not both.
        """
        if files is None and texts is None:
            raise ValueError("Either 'files' or 'texts' must be provided")
        if files is not None and texts is not None:
            raise ValueError("Cannot provide both 'files' and 'texts'. Use one or the other.")

        documents: list[VectorDocument] = []

        if texts is not None:
            if metadatas is None:
                metadatas = [{}] * len(texts)
            if ids is None:
                ids = [hashlib.md5(text.encode()).hexdigest() for text in texts]

            if len(metadatas) < len(texts):
                metadatas = list(metadatas) + [{}] * (len(texts) - len(metadatas))
            if len(ids) < len(texts):
                ids = list(ids) + [str(uuid.uuid4()) for _ in range(len(texts) - len(ids))]

            for text, metadata, doc_id in zip(texts, metadatas, ids, strict=False):
                documents.append(VectorDocument(id=doc_id, content=text, metadata=metadata))
        elif files is not None:
            documents = await self._document_loader.load_files(
                files=files,
                metadatas=metadatas,
                ids=ids,
                chunking_strategy=chunking_strategy,
                chunking_params=chunking_params,
                embedder=self.embedder,
            )

        return await self.add_documents(documents)

    async def add_documents(self, documents: list[VectorDocument]) -> bool:
        """Add documents to the vector database with batch embedding"""
        if self.embedder:
            docs_to_embed = [doc for doc in documents if doc.vector is None]
            if docs_to_embed:
                contents = [doc.content for doc in docs_to_embed]
                embeddings = await self.embedder.embed_batch(contents)
                for doc, embedding in zip(docs_to_embed, embeddings, strict=False):
                    doc.vector = embedding

        return await self._add_documents_impl(documents)

    def as_retriever(self, **kwargs):
        """Create a retriever that can be called directly like LangChain

        Args:
            **kwargs: Any parameters supported by the specific vector database
                     Common parameters:
                     - k: Number of documents to retrieve (default: 5)
                     - score_threshold: Minimum similarity score
                     - filter: Metadata filters
                     - search_type: Type of search (similarity, mmr, etc.)
                     - Any other database-specific parameters
        """
        return self._create_retriever(**kwargs)

    def _create_retriever(self, **kwargs):
        """Create a retriever that can be called directly like LangChain"""

        class VectorRetriever:
            def __init__(self, vector_db, **search_kwargs):
                self.vector_db = vector_db
                self.search_kwargs = search_kwargs
                self.top_k = search_kwargs.get("k", 5)
                self.score_threshold = search_kwargs.get("score_threshold")
                self.filter = search_kwargs.get("filter")
                self.search_type = search_kwargs.get("search_type", "similarity")

            async def async_search(self, query: str):
                """Search for documents - async method"""
                results = await self.vector_db._async_search_impl(query, top_k=self.top_k)
                return self._convert_results_to_docs(results)

            def search(self, query: str):
                """Search for documents - sync method"""
                results = self.vector_db.search(query, top_k=self.top_k)
                return self._convert_results_to_docs(results)

            def _convert_results_to_docs(self, results):
                """Convert search results to document format - shared by async and sync"""
                if self.score_threshold is not None:
                    results = [r for r in results if r.score >= self.score_threshold]

                docs = []
                for result in results:

                    class Document:
                        def __init__(self, content, metadata, score):
                            self.page_content = content
                            self.metadata = metadata
                            self.score = score

                        def __repr__(self):
                            return f"Document(score={self.score:.3f}, content={self.page_content!r}, metadata={self.metadata})"

                    doc = Document(result.content, result.metadata, result.score)
                    docs.append(doc)
                return docs

            def __repr__(self):
                return f"VectorRetriever(k={self.top_k}, vector_db={type(self.vector_db).__name__})"

        return VectorRetriever(self, **kwargs)

    async def _async_search_impl(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
        """Internal async search implementation"""
        if self.embedder:
            query_vector = await self.embedder.embed(query)
            return await self._search_impl(query_vector, top_k)
        else:
            return await self._search_text_impl(query, top_k)

    async def async_delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        """Delete documents by IDs or where filter"""
        return await self._delete_impl(ids, where)

    async def async_update(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        """Update document metadata, content, or embeddings"""
        return await self._update_impl(ids, metadatas, documents, embeddings)

    async def async_upsert(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        """Upsert documents (update if exists, add if not)"""
        return await self._upsert_impl(ids, metadatas, documents, embeddings)

    @abstractmethod
    async def _add_documents_impl(self, documents: list[VectorDocument]) -> bool:
        """Implementation-specific document addition"""
        pass

    @abstractmethod
    async def _search_impl(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Implementation-specific vector search"""
        pass

    @abstractmethod
    async def _search_text_impl(self, query: str, top_k: int) -> list[VectorSearchResult]:
        """Implementation-specific text search"""
        pass

    @abstractmethod
    async def _delete_impl(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        """Implementation-specific document deletion"""
        pass

    @abstractmethod
    async def _update_impl(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        """Implementation-specific document update"""
        pass

    @abstractmethod
    async def _upsert_impl(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        """Implementation-specific document upsert"""
        pass

    async def async_get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get documents by IDs or filters (async)"""
        return await self._get_documents_impl(ids, where, limit, include)

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get documents by IDs or filters (sync)"""
        return self._get_documents_impl_sync(ids, where, limit, include)

    @abstractmethod
    async def _get_documents_impl(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Implementation-specific document retrieval (async)"""
        pass

    @abstractmethod
    def _get_documents_impl_sync(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Implementation-specific document retrieval (sync)"""
        pass

    async def async_count(self) -> int:
        """Get total document count (async)"""
        return await self._count_documents_impl()

    def count(self) -> int:
        """Get total document count (sync)"""
        return self._count_documents_impl_sync()

    @abstractmethod
    async def _count_documents_impl(self) -> int:
        """Implementation-specific document counting (async)"""
        pass

    @abstractmethod
    def _count_documents_impl_sync(self) -> int:
        """Implementation-specific document counting (sync)"""
        pass

    def initialize_sync(self) -> bool:
        """Initialize the vector database and embedder synchronously"""
        if self.embedder:
            self.embedder.initialize_sync()
        return self._initialize_db_sync()

    @abstractmethod
    def _initialize_db_sync(self) -> bool:
        """Initialize the specific vector database synchronously"""
        pass

    def add(
        self,
        files: list[str] | None = None,
        texts: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        chunking_strategy: str = "document",
        chunking_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> bool:
        """Add files or text to the vector database with automatic processing (sync)

        Args:
            files: List of file paths or URLs to load
            texts: List of text strings to add directly (alternative to files)
            metadatas: Optional list of metadata dicts (one per file/text)
            ids: Optional list of document IDs (one per file/text)
            chunking_strategy: Chunking strategy to use (only for files)
            chunking_params: Optional parameters for chunking
            **kwargs: Additional parameters

        Note: Either files or texts must be provided, not both.
        """
        if files is None and texts is None:
            raise ValueError("Either 'files' or 'texts' must be provided")
        if files is not None and texts is not None:
            raise ValueError("Cannot provide both 'files' and 'texts'. Use one or the other.")

        documents: list[VectorDocument] = []

        if texts is not None:
            if metadatas is None:
                metadatas = [{}] * len(texts)
            if ids is None:
                ids = [hashlib.md5(text.encode()).hexdigest() for text in texts]

            if len(metadatas) < len(texts):
                metadatas = list(metadatas) + [{}] * (len(texts) - len(metadatas))
            if len(ids) < len(texts):
                ids = list(ids) + [str(uuid.uuid4()) for _ in range(len(texts) - len(ids))]

            for text, metadata, doc_id in zip(texts, metadatas, ids, strict=False):
                documents.append(VectorDocument(id=doc_id, content=text, metadata=metadata))
        elif files is not None:
            documents = self._document_loader.load_files_sync(
                files=files,
                metadatas=metadatas,
                ids=ids,
                chunking_strategy=chunking_strategy,
                chunking_params=chunking_params,
                embedder=self.embedder,
            )

        return self.add_documents_sync(documents)

    def add_documents_sync(self, documents: list[VectorDocument]) -> bool:
        """Add documents to the vector database with batch embedding (sync)"""
        if self.embedder:
            docs_to_embed = [doc for doc in documents if doc.vector is None]
            if docs_to_embed:
                contents = [doc.content for doc in docs_to_embed]
                embeddings = self.embedder.embed_batch_sync(contents)
                for doc, embedding in zip(docs_to_embed, embeddings, strict=False):
                    doc.vector = embedding

        return self._add_documents_impl_sync(documents)

    def search(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
        """Search for documents synchronously"""
        if self.embedder:
            query_vector = self.embedder.embed_sync(query)
            return self._search_impl_sync(query_vector, top_k)
        else:
            return self._search_text_impl_sync(query, top_k)

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        """Delete documents by IDs or where filter (sync)"""
        return self._delete_impl_sync(ids, where)

    def update(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        """Update document metadata, content, or embeddings (sync)"""
        return self._update_impl_sync(ids, metadatas, documents, embeddings)

    def upsert(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        """Upsert documents (update if exists, add if not) (sync)"""
        return self._upsert_impl_sync(ids, metadatas, documents, embeddings)

    @abstractmethod
    def _add_documents_impl_sync(self, documents: list[VectorDocument]) -> bool:
        """Implementation-specific document addition (sync)"""
        pass

    @abstractmethod
    def _search_impl_sync(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
        """Implementation-specific vector search (sync)"""
        pass

    @abstractmethod
    def _search_text_impl_sync(self, query: str, top_k: int) -> list[VectorSearchResult]:
        """Implementation-specific text search (sync)"""
        pass

    @abstractmethod
    def _delete_impl_sync(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        """Implementation-specific document deletion (sync)"""
        pass

    @abstractmethod
    def _update_impl_sync(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        """Implementation-specific document update (sync)"""
        pass

    @abstractmethod
    def _upsert_impl_sync(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        """Implementation-specific document upsert (sync)"""
        pass
