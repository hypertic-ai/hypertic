from dataclasses import dataclass
from typing import Any

from hypertic.vectordb.chunking.base import Chunk, ChunkingStrategy


@dataclass
class DocumentChunker(ChunkingStrategy):
    chunk_size: int = 2000
    chunk_overlap: int = 200
    separators: list[str] | None = None
    keep_separator: bool = True
    add_start_index: bool = False

    def __post_init__(self):
        if self.separators is None:
            self.separators = [
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ]

    def _get_chunk_type(self) -> str:
        return "document"

    def _chunk_impl_async(self, text: str, params: dict[str, Any]) -> list[Chunk]:
        chunks = self._split_by_document_structure(
            text,
            params["chunk_size"],
            params["chunk_overlap"],
            params["separators"],
            params["keep_separator"],
        )

        result = self._create_chunks_from_texts(
            chunks,
            self._get_chunk_type(),
            params["chunk_size"],
            params["chunk_overlap"],
            params["add_start_index"],
        )

        for i, chunk in enumerate(result):
            if i > 0:
                chunk.overlap_with_previous = True
            if i < len(result) - 1:
                chunk.overlap_with_next = True

        return result

    def _chunk_impl_sync(self, text: str, params: dict[str, Any]) -> list[Chunk]:
        chunks = self._split_by_document_structure(
            text,
            params["chunk_size"],
            params["chunk_overlap"],
            params["separators"],
            params["keep_separator"],
        )

        result = self._create_chunks_from_texts(
            chunks,
            self._get_chunk_type(),
            params["chunk_size"],
            params["chunk_overlap"],
            params["add_start_index"],
        )

        for i, chunk in enumerate(result):
            if i > 0:
                chunk.overlap_with_previous = True
            if i < len(result) - 1:
                chunk.overlap_with_next = True

        return result

    def _split_by_document_structure(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str],
        keep_separator: bool,
    ) -> list[str]:
        for separator in separators:
            if separator:
                splits = self._split_text_on_separator(text, separator, keep_separator)
            else:
                splits = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

            if all(len(split) <= chunk_size for split in splits):
                return self._merge_splits(splits, chunk_size, chunk_overlap)

        return self._split_by_characters(text, chunk_size, chunk_overlap)

    def _split_text_on_separator(self, text: str, separator: str, keep_separator: bool) -> list[str]:
        if separator == "":
            return list(text)

        splits = text.split(separator)

        if keep_separator:
            result = []
            for i, split in enumerate(splits):
                if i < len(splits) - 1:
                    result.append(split + separator)
                else:
                    result.append(split)
            return result
        else:
            return splits

    def _merge_splits(self, splits: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
        if not splits:
            return []

        chunks = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(split) > chunk_size and current_chunk:
                chunks.append(current_chunk)

                if chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + split
                else:
                    current_chunk = split
            else:
                current_chunk += split

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_by_characters(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)

            start = end - chunk_overlap
            if start <= 0:
                start = end

        return chunks


@dataclass
class MarkdownChunker(ChunkingStrategy):
    chunk_size: int = 2000
    chunk_overlap: int = 200
    headers_to_split_on: list[tuple[str, str]] | None = None
    strip_headers: bool = True
    add_start_index: bool = False

    def __post_init__(self):
        if self.headers_to_split_on is None:
            self.headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
                ("######", "Header 6"),
            ]

        self.headers_to_split_on = sorted(self.headers_to_split_on, key=lambda x: len(x[0]), reverse=True)

    def _get_chunk_type(self) -> str:
        """Get the chunk type for this strategy"""
        return "markdown"

    def _chunk_impl_async(self, text: str, params: dict[str, Any]) -> list[Chunk]:
        chunks = self._split_markdown_by_structure(text, params["chunk_size"], params["chunk_overlap"], params["strip_headers"])

        result = self._create_chunks_from_tuples(
            chunks,
            self._get_chunk_type(),
            params["chunk_size"],
            params["chunk_overlap"],
            params["add_start_index"],
        )

        for i, chunk in enumerate(result):
            if i > 0:
                chunk.overlap_with_previous = True
            if i < len(result) - 1:
                chunk.overlap_with_next = True

        return result

    def _chunk_impl_sync(self, text: str, params: dict[str, Any]) -> list[Chunk]:
        chunks = self._split_markdown_by_structure(text, params["chunk_size"], params["chunk_overlap"], params["strip_headers"])

        result = self._create_chunks_from_tuples(
            chunks,
            self._get_chunk_type(),
            params["chunk_size"],
            params["chunk_overlap"],
            params["add_start_index"],
        )

        for i, chunk in enumerate(result):
            if i > 0:
                chunk.overlap_with_previous = True
            if i < len(result) - 1:
                chunk.overlap_with_next = True

        return result

    def _split_markdown_by_structure(self, text: str, chunk_size: int, chunk_overlap: int, strip_headers: bool) -> list[tuple[str, dict[str, Any]]]:
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_metadata: dict[str, Any] = {}
        header_stack: list[tuple[int, str, str]] = []

        in_code_block = False
        opening_fence = ""

        for line in lines:
            stripped_line = line.strip()

            if not in_code_block:
                if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                    in_code_block = True
                    opening_fence = "```"
                elif stripped_line.startswith("~~~"):
                    in_code_block = True
                    opening_fence = "~~~"
            elif stripped_line.startswith(opening_fence):
                in_code_block = False
                opening_fence = ""

            if in_code_block:
                current_chunk.append(line)
                continue

            header_found = False
            if self.headers_to_split_on is not None:
                for header_pattern, metadata_key in self.headers_to_split_on:
                    if self._is_header_line(stripped_line, header_pattern):
                        if current_chunk:
                            chunk_text = "\n".join(current_chunk)
                            if len(chunk_text.strip()) > 0:
                                chunks.append((chunk_text, current_metadata.copy()))
                            current_chunk = []

                        header_text = stripped_line[len(header_pattern) :].strip()

                        header_level = len(header_pattern)
                        self._update_header_stack(header_stack, header_level, metadata_key, header_text)

                        current_metadata = {h[1]: h[2] for h in header_stack}

                        if not strip_headers:
                            current_chunk.append(line)

                        header_found = True
                        break

            if not header_found:
                current_chunk.append(line)

                current_text = "\n".join(current_chunk)
                if len(current_text) > chunk_size:
                    if current_chunk:
                        chunk_text = "\n".join(current_chunk)
                        if len(chunk_text.strip()) > 0:
                            chunks.append((chunk_text, current_metadata.copy()))
                        current_chunk = []

        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            if len(chunk_text.strip()) > 0:
                chunks.append((chunk_text, current_metadata.copy()))

        if chunk_overlap > 0:
            chunks = self._apply_overlap(chunks, chunk_overlap)

        return chunks

    def _is_header_line(self, line: str, header_pattern: str) -> bool:
        return line.startswith(header_pattern) and (len(line) == len(header_pattern) or line[len(header_pattern)] == " ")

    def _update_header_stack(self, header_stack: list[tuple[int, str, str]], level: int, key: str, text: str):
        while header_stack and header_stack[-1][0] >= level:
            header_stack.pop()

        header_stack.append((level, key, text))

    def _apply_overlap(self, chunks: list[tuple[str, dict[str, Any]]], overlap: int) -> list[tuple[str, dict[str, Any]]]:
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = []

        for i, (content, metadata) in enumerate(chunks):
            if i > 0 and overlap > 0:
                prev_content = chunks[i - 1][0]
                overlap_text = prev_content[-overlap:] if len(prev_content) > overlap else prev_content
                overlapped_content = overlap_text + content
                overlapped_chunks.append((overlapped_content, metadata))
            else:
                overlapped_chunks.append((content, metadata))

        return overlapped_chunks


@dataclass
class SemanticChunker(ChunkingStrategy):
    embedder: Any
    chunk_size: int = 2000
    threshold: float = 0.5
    min_sentences_per_chunk: int = 1
    add_start_index: bool = False
    chunk_overlap: int = 0

    def _get_chunk_type(self) -> str:
        return "semantic"

    def _chunk_impl_async(self, text: str, params: dict[str, Any]) -> list[Chunk]:
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return self._create_single_chunk(text, self._get_chunk_type())

        breakpoints = self._find_semantic_breakpoints_sync(sentences, params["threshold"], params["min_sentences_per_chunk"])

        chunks = self._create_chunks_from_breakpoints(sentences, breakpoints, params["chunk_size"])

        result = self._create_chunks_from_texts(
            chunks,
            self._get_chunk_type(),
            params["chunk_size"],
            params["chunk_overlap"],
            params["add_start_index"],
        )

        for i, chunk in enumerate(result):
            if i > 0:
                chunk.overlap_with_previous = False
            if i < len(result) - 1:
                chunk.overlap_with_next = False

        return result

    def _chunk_impl_sync(self, text: str, params: dict[str, Any]) -> list[Chunk]:
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return self._create_single_chunk(text, self._get_chunk_type())

        breakpoints = self._find_semantic_breakpoints_sync(sentences, params["threshold"], params["min_sentences_per_chunk"])

        chunks = self._create_chunks_from_breakpoints(sentences, breakpoints, params["chunk_size"])

        result = self._create_chunks_from_texts(
            chunks,
            self._get_chunk_type(),
            params["chunk_size"],
            params["chunk_overlap"],
            params["add_start_index"],
        )

        for i, chunk in enumerate(result):
            if i > 0:
                chunk.overlap_with_previous = False
            if i < len(result) - 1:
                chunk.overlap_with_next = False

        return result

    def _split_into_sentences(self, text: str) -> list[str]:
        import re

        sentence_endings = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    async def _find_semantic_breakpoints(self, sentences: list[str], threshold: float, min_sentences_per_chunk: int) -> list[int]:
        if len(sentences) < 2:
            return []

        embeddings = await self.embedder.embed_batch(sentences)

        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(similarity)

        breakpoints = []
        for i, similarity in enumerate(similarities):
            if similarity < threshold:
                if i + 1 >= min_sentences_per_chunk:
                    breakpoints.append(i + 1)

        return breakpoints

    def _find_semantic_breakpoints_sync(self, sentences: list[str], threshold: float, min_sentences_per_chunk: int) -> list[int]:
        if len(sentences) < 2:
            return []

        embeddings = self.embedder.embed_batch_sync(sentences)

        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(similarity)

        breakpoints = []
        for i, similarity in enumerate(similarities):
            if similarity < threshold:
                if i + 1 >= min_sentences_per_chunk:
                    breakpoints.append(i + 1)

        return breakpoints

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        import numpy as np

        a = np.array(vec1)
        b = np.array(vec2)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        result = dot_product / (norm_a * norm_b)
        return float(result)

    def _create_chunks_from_breakpoints(self, sentences: list[str], breakpoints: list[int], chunk_size: int) -> list[str]:
        chunks = []
        start_idx = 0

        all_breakpoints = sorted([*breakpoints, len(sentences)])

        for end_idx in all_breakpoints:
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            if chunk_size and len(chunk_text) > chunk_size:
                current_chunk = ""
                for sentence in chunk_sentences:
                    if len(current_chunk + " " + sentence) <= chunk_size:
                        current_chunk += (" " + sentence) if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence

                if current_chunk:
                    chunks.append(current_chunk)
            else:
                chunks.append(chunk_text)

            start_idx = end_idx

        return chunks


def get_chunker(strategy: str, embedder=None, **kwargs):
    if strategy == "document":
        return DocumentChunker(**kwargs)
    elif strategy == "markdown":
        return MarkdownChunker(**kwargs)
    elif strategy == "semantic":
        return SemanticChunker(embedder=embedder, **kwargs)
    elif strategy == "none":
        return DocumentChunker(chunk_size=2**31 - 1, chunk_overlap=0)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def get_chunker_sync(strategy: str, embedder=None, **kwargs):
    if strategy == "document":
        return DocumentChunker(**kwargs)
    elif strategy == "markdown":
        return MarkdownChunker(**kwargs)
    elif strategy == "semantic":
        return SemanticChunker(embedder=embedder, **kwargs)
    elif strategy == "none":
        return DocumentChunker(chunk_size=2**31 - 1, chunk_overlap=0)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
