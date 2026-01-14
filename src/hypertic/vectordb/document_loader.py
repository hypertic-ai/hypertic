import hashlib
import os
import uuid
from typing import Any
from urllib.parse import urlparse

from hypertic.utils.document_processor import DocumentProcessor
from hypertic.utils.log import get_logger
from hypertic.vectordb.chunking.strategies import get_chunker, get_chunker_sync

logger = get_logger(__name__)


class DocumentLoader:
    """Internal class for loading files and converting them to VectorDocuments"""

    def __init__(self, processor: DocumentProcessor | None = None):
        self.processor = processor or DocumentProcessor()

    def _get_file_type(self, file_path: str) -> str:
        if file_path.startswith(("http://", "https://")):
            parsed = urlparse(file_path)
            path = parsed.path
        else:
            path = file_path
        return os.path.splitext(path)[1].lower()

    def _prepare_file_metadata(self, files: list[str], metadatas: list[dict[str, Any]] | None, ids: list[str] | None):
        if metadatas is None:
            metadatas = [{}] * len(files)
        if ids is None:
            ids = []
            for file_path in files:
                try:
                    if file_path.startswith(("http://", "https://")):
                        file_id = hashlib.md5(file_path.encode()).hexdigest()
                    else:
                        stat = os.stat(file_path)
                        file_id = hashlib.md5(f"{file_path}_{stat.st_mtime}_{stat.st_size}".encode()).hexdigest()
                    ids.append(file_id)
                except Exception:
                    ids.append(str(uuid.uuid4()))
        return metadatas, ids

    def _create_document_from_chunk(
        self,
        chunk,
        doc_id: str,
        metadata: dict[str, Any],
        file_path: str,
        chunk_index: int,
        total_chunks: int,
    ):
        from hypertic.vectordb.base import VectorDocument

        if total_chunks > 1:
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{chunk_index}"))
        else:
            chunk_id = doc_id

        chunk_metadata = {
            **metadata,
            "source": file_path,
            "file_type": self._get_file_type(file_path),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "original_doc_id": doc_id,
            **chunk.metadata,
        }

        return VectorDocument(id=chunk_id, content=chunk.content, metadata=chunk_metadata)

    def _create_document_from_content(self, content: str, doc_id: str, metadata: dict[str, Any], file_path: str):
        from hypertic.vectordb.base import VectorDocument

        return VectorDocument(
            id=doc_id,
            content=content,
            metadata={**metadata, "source": file_path, "file_type": self._get_file_type(file_path)},
        )

    async def load_files(
        self,
        files: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        chunking_strategy: str = "document",
        chunking_params: dict[str, Any] | None = None,
        embedder=None,
    ):
        metadatas, ids = self._prepare_file_metadata(files, metadatas, ids)
        documents = []

        for file_path, metadata, doc_id in zip(files, metadatas, ids, strict=False):
            try:
                content = await self.processor.process_file(file_path)

                if chunking_strategy and chunking_strategy != "none":
                    chunker = get_chunker(chunking_strategy, embedder, **(chunking_params or {}))
                    chunks = await chunker.chunk(content)

                    for i, chunk in enumerate(chunks):
                        doc = self._create_document_from_chunk(chunk, doc_id, metadata, file_path, i, len(chunks))
                        documents.append(doc)
                else:
                    doc = self._create_document_from_content(content, doc_id, metadata, file_path)
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                continue

        return documents

    def load_files_sync(
        self,
        files: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        chunking_strategy: str = "document",
        chunking_params: dict[str, Any] | None = None,
        embedder=None,
    ):
        metadatas, ids = self._prepare_file_metadata(files, metadatas, ids)
        documents = []

        for file_path, metadata, doc_id in zip(files, metadatas, ids, strict=False):
            try:
                content = self.processor.process_file_sync(file_path)

                if chunking_strategy and chunking_strategy != "none":
                    chunker = get_chunker_sync(chunking_strategy, embedder, **(chunking_params or {}))
                    chunks = chunker.chunk_sync(content)

                    for i, chunk in enumerate(chunks):
                        doc = self._create_document_from_chunk(chunk, doc_id, metadata, file_path, i, len(chunks))
                        documents.append(doc)
                else:
                    doc = self._create_document_from_content(content, doc_id, metadata, file_path)
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
                continue

        return documents
