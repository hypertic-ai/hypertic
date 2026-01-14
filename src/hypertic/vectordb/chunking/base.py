from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any] | None = None
    chunk_index: int = 0
    total_chunks: int = 1
    chunk_type: str = "default"
    overlap_with_previous: bool = False
    overlap_with_next: bool = False

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ChunkingStrategy(ABC):
    def _extract_parameters(self, kwargs: dict[str, Any], chunk_type: str) -> dict[str, Any]:
        params = {
            "chunk_size": kwargs.get("chunk_size", getattr(self, "chunk_size", 1000)),
            "chunk_overlap": kwargs.get("chunk_overlap", getattr(self, "chunk_overlap", 200)),
            "add_start_index": kwargs.get("add_start_index", getattr(self, "add_start_index", False)),
        }

        if hasattr(self, "separators"):
            params["separators"] = kwargs.get("separators", self.separators)
        if hasattr(self, "keep_separator"):
            params["keep_separator"] = kwargs.get("keep_separator", self.keep_separator)
        if hasattr(self, "strip_headers"):
            params["strip_headers"] = kwargs.get("strip_headers", self.strip_headers)
        if hasattr(self, "threshold"):
            params["threshold"] = kwargs.get("threshold", self.threshold)
        if hasattr(self, "min_sentences_per_chunk"):
            params["min_sentences_per_chunk"] = kwargs.get("min_sentences_per_chunk", self.min_sentences_per_chunk)

        return params

    def _create_single_chunk(self, text: str, chunk_type: str) -> list[Chunk]:
        return [self._create_chunk(content=text, chunk_index=0, total_chunks=1, chunk_type=chunk_type)]

    def _create_chunks_from_texts(
        self,
        chunk_texts: list[str],
        chunk_type: str,
        chunk_size: int,
        chunk_overlap: int,
        add_start_index: bool,
    ) -> list[Chunk]:
        result = []
        for i, chunk_text in enumerate(chunk_texts):
            metadata = {}
            if add_start_index:
                start_index = i * (chunk_size - chunk_overlap)
                metadata["start_index"] = start_index

            result.append(
                self._create_chunk(
                    content=chunk_text,
                    chunk_index=i,
                    total_chunks=len(chunk_texts),
                    chunk_type=chunk_type,
                    metadata=metadata,
                )
            )
        return result

    def _create_chunks_from_tuples(
        self,
        chunk_tuples: list[tuple[str, dict[str, Any]]],
        chunk_type: str,
        chunk_size: int,
        chunk_overlap: int,
        add_start_index: bool,
    ) -> list[Chunk]:
        result = []
        for i, (chunk_text, metadata) in enumerate(chunk_tuples):
            chunk_metadata = metadata.copy()
            if add_start_index:
                chunk_metadata["start_index"] = i * (chunk_size - chunk_overlap)

            result.append(
                self._create_chunk(
                    content=chunk_text,
                    chunk_index=i,
                    total_chunks=len(chunk_tuples),
                    chunk_type=chunk_type,
                    metadata=chunk_metadata,
                )
            )
        return result

    def chunk_unified(self, text: str, is_async: bool = True, **kwargs) -> list[Chunk]:
        """
        Args:
            text: The text to chunk
            is_async: Whether to use async or sync implementation
            **kwargs: Additional parameters for the chunking strategy

        Returns:
            List of Chunk objects
        """
        params = self._extract_parameters(kwargs, self._get_chunk_type())

        if len(text) <= params["chunk_size"]:
            return self._create_single_chunk(text, self._get_chunk_type())

        if is_async:
            return self._chunk_impl_async(text, params)
        else:
            return self._chunk_impl_sync(text, params)

    @abstractmethod
    def _get_chunk_type(self) -> str:
        """Get the chunk type for this strategy"""
        pass

    @abstractmethod
    def _chunk_impl_async(self, text: str, params: dict[str, Any]) -> list[Chunk]:
        """Async implementation of chunking logic"""
        pass

    @abstractmethod
    def _chunk_impl_sync(self, text: str, params: dict[str, Any]) -> list[Chunk]:
        """Sync implementation of chunking logic"""
        pass

    async def chunk(self, text: str, **kwargs) -> list[Chunk]:
        """Split text into chunks (async)"""
        return self.chunk_unified(text, is_async=True, **kwargs)

    def chunk_sync(self, text: str, **kwargs) -> list[Chunk]:
        """Split text into chunks (sync)"""
        return self.chunk_unified(text, is_async=False, **kwargs)

    def _create_chunk(
        self,
        content: str,
        chunk_index: int,
        total_chunks: int,
        chunk_type: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> Chunk:
        """Helper method to create a Chunk object with proper metadata"""
        if metadata is None:
            metadata = {}

        metadata.update({"chunk_index": chunk_index, "total_chunks": total_chunks, "chunk_type": chunk_type})

        return Chunk(
            content=content,
            metadata=metadata,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            chunk_type=chunk_type,
        )
