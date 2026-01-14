import asyncio
import warnings
from dataclasses import dataclass, field
from math import sqrt
from os import getenv
from typing import Any

from hypertic.utils.log import get_logger
from hypertic.vectordb.base import BaseVectorDB, VectorDocument, VectorSearchResult

warnings.filterwarnings("ignore", category=Warning, module="sqlalchemy")
warnings.filterwarnings("ignore", category=Warning, module="psycopg2")


def clean_error_message(error_msg: str) -> str:
    if "(Background on this error at:" in error_msg:
        return error_msg.split("(Background on this error at:")[0].strip()
    return error_msg


logger = get_logger(__name__)

try:
    from sqlalchemy import delete, func, select, text, update
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.engine import Engine, create_engine
    from sqlalchemy.inspection import inspect
    from sqlalchemy.orm import Session, scoped_session, sessionmaker
    from sqlalchemy.schema import Column, MetaData, Table
    from sqlalchemy.types import DateTime, String, Text
except ImportError as err:
    raise ImportError("SQLAlchemy required for PgVectorDB. Install with: pip install sqlalchemy") from err

try:
    from pgvector.sqlalchemy import Vector
except ImportError as err:
    raise ImportError("pgvector required for PgVectorDB. Install with: pip install pgvector") from err


class Distance:
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"


class SearchType:
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class HNSW:
    m: int = 16
    ef_construction: int = 64
    ef_search: int = 40
    name: str | None = None


@dataclass
class Ivfflat:
    lists: int = 100
    probes: int = 1
    name: str | None = None
    dynamic_lists: bool = True


@dataclass
class PgVectorDB(BaseVectorDB):
    collection: str
    db_url: str | None = None
    db_engine: Engine | None = None
    embedder: Any | None = None
    schema: str = "ai"
    vector_size: int = 1536
    distance_metric: str = Distance.COSINE
    search_type: str = SearchType.VECTOR
    vector_index: Ivfflat | HNSW = field(default_factory=HNSW)
    prefix_match: bool = False
    vector_score_weight: float = 0.5
    content_language: str = "english"

    table_name: str = field(init=False)

    _pool: Any | None = field(default=None, init=False)
    _sync_conn: Any | None = field(default=None, init=False)

    metadata: MetaData | None = field(default=None, init=False)
    _Session: Any | None = field(default=None, init=False)
    table: Any | None = field(default=None, init=False)

    def __post_init__(self):
        if not self.collection:
            raise ValueError("Collection name must be provided.")

        if self.db_url is None:
            self.db_url = getenv("PGVECTORDB_URL")

        if self.db_engine is None and self.db_url is None:
            raise ValueError("Either 'db_url' or 'db_engine' must be provided. Set PGVECTORDB_URL environment variable or pass db_url parameter.")

        BaseVectorDB.__init__(self, embedder=self.embedder)

        self.table_name = self.collection

        if self.db_engine is None:
            if self.db_url is None:
                raise ValueError("Must provide 'db_url' if 'db_engine' is None. Set PGVECTORDB_URL environment variable or pass db_url parameter.")
            try:
                self.db_engine = create_engine(self.db_url)
            except Exception as e:
                error_msg = str(e)
                if "psycopg2" in error_msg.lower() or "No module named 'psycopg2" in error_msg:
                    raise ImportError("Install with: pip install hypertic[pgvector] or pip install psycopg2-binary") from e
                logger.error(f"Failed to create engine from 'db_url': {e}", exc_info=True)
                raise

        self.metadata = MetaData(schema=self.schema)
        self._Session = scoped_session(sessionmaker(bind=self.db_engine))
        self.table = self._get_table()

    def _get_table(self):
        if self.metadata is None:
            raise RuntimeError("Metadata not initialized")
        return Table(
            self.table_name,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("name", String),
            Column("meta_data", postgresql.JSONB, server_default=text("'{}'::jsonb")),
            Column("filters", postgresql.JSONB, server_default=text("'{}'::jsonb"), nullable=True),
            Column("content", Text),
            Column("embedding", Vector(self.vector_size)),
            Column("usage", postgresql.JSONB, server_default=text("'{}'::jsonb")),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
            Column("updated_at", DateTime(timezone=True), onupdate=func.now()),
            Column("content_hash", String),
            Column("content_id", String),
            extend_existing=True,
        )

    def table_exists(self) -> bool:
        try:
            if self.db_engine is None:
                return False
            inspector = inspect(self.db_engine)
            if inspector is None:
                return False
            return bool(inspector.has_table(self.table_name, schema=self.schema))
        except Exception as e:
            logger.error(f"Error checking if table exists: {e}", exc_info=True)
            return False

    async def _initialize_db(self) -> bool:
        try:
            if self.table_exists():
                return True

            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                sess.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                if self.schema is not None:
                    sess.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema};"))

            if self.table is None:
                raise RuntimeError("Table not initialized")
            self.table.create(self.db_engine)

            self._create_vector_index()

            self._create_gin_index()

            return True

        except Exception as e:
            logger.error(f"Error initializing PgVector database: {e}", exc_info=True)
            return False

    def _initialize_db_sync(self) -> bool:
        try:
            if self.table_exists():
                return True

            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                sess.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                if self.schema is not None:
                    sess.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema};"))

            if self.table is None:
                raise RuntimeError("Table not initialized")
            self.table.create(self.db_engine)

            self._create_vector_index()
            self._create_gin_index()

            return True

        except Exception as e:
            logger.error(f"Error initializing PgVector database: {e}", exc_info=True)
            return False

    def _create_vector_index(self, force_recreate: bool = False) -> None:
        if self.vector_index is None:
            return

        if self.vector_index.name is None:
            index_type = "ivfflat" if isinstance(self.vector_index, Ivfflat) else "hnsw"
            self.vector_index.name = f"{self.table_name}_{index_type}_index"

        index_distance = {
            Distance.L2: "vector_l2_ops",
            Distance.INNER_PRODUCT: "vector_ip_ops",
            Distance.COSINE: "vector_cosine_ops",
        }.get(self.distance_metric, "vector_cosine_ops")

        vector_index_exists = self._index_exists(self.vector_index.name)

        if vector_index_exists and not force_recreate:
            return

        if vector_index_exists and force_recreate:
            self._drop_index(self.vector_index.name)

        try:
            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                if isinstance(self.vector_index, Ivfflat):
                    self._create_ivfflat_index(sess, index_distance)
                elif isinstance(self.vector_index, HNSW):
                    self._create_hnsw_index(sess, index_distance)
        except Exception as e:
            logger.error(f"Error creating vector index '{self.vector_index.name}': {e}", exc_info=True)
            raise

    def _create_ivfflat_index(self, sess: Session, index_distance: str) -> None:
        if isinstance(self.vector_index, Ivfflat):
            num_lists = self.vector_index.lists
            if self.vector_index.dynamic_lists:
                total_records = self._count_documents_impl_sync()
                if total_records < 1000000:
                    num_lists = max(int(total_records / 1000), 1)
                else:
                    num_lists = max(int(sqrt(total_records)), 1)

            sess.execute(text("SET ivfflat.probes = :probes;"), {"probes": self.vector_index.probes})

            if self.table is None:
                raise RuntimeError("Table not initialized")
            index_name = self.vector_index.name or f"{self.table_name}_ivfflat_index"
            create_index_sql = text(
                f'CREATE INDEX "{index_name}" ON {self.table.fullname} USING ivfflat (embedding {index_distance}) WITH (lists = :num_lists);'
            )
            sess.execute(create_index_sql, {"num_lists": num_lists})
        else:
            raise ValueError("vector_index must be Ivfflat for IVFFlat index creation")

    def _create_hnsw_index(self, sess: Session, index_distance: str) -> None:
        if isinstance(self.vector_index, HNSW):
            if self.table is None:
                raise RuntimeError("Table not initialized")
            index_name = self.vector_index.name or f"{self.table_name}_hnsw_index"
            create_index_sql = text(
                f'CREATE INDEX "{index_name}" ON {self.table.fullname} '
                f"USING hnsw (embedding {index_distance}) "
                f"WITH (m = :m, ef_construction = :ef_construction);"
            )
            sess.execute(
                create_index_sql,
                {"m": self.vector_index.m, "ef_construction": self.vector_index.ef_construction},
            )
        else:
            raise ValueError("vector_index must be HNSW for HNSW index creation")

    def _create_gin_index(self, force_recreate: bool = False) -> None:
        gin_index_name = f"{self.table_name}_content_gin_index"

        gin_index_exists = self._index_exists(gin_index_name)

        if gin_index_exists and not force_recreate:
            return

        if gin_index_exists and force_recreate:
            self._drop_index(gin_index_name)

        try:
            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                if self.table is None:
                    raise RuntimeError("Table not initialized")
                create_gin_index_sql = text(
                    f"CREATE INDEX \"{gin_index_name}\" ON {self.table.fullname} USING GIN (to_tsvector('{self.content_language}', content));"
                )
                sess.execute(create_gin_index_sql)
        except Exception as e:
            logger.error(f"Error creating GIN index '{gin_index_name}': {e}", exc_info=True)
            raise

    def _index_exists(self, index_name: str) -> bool:
        if self.db_engine is None:
            return False
        inspector = inspect(self.db_engine)
        if inspector is None:
            return False
        if self.table is None:
            raise RuntimeError("Table not initialized")
        indexes = inspector.get_indexes(self.table.name, schema=self.schema)
        return any(idx["name"] == index_name for idx in indexes)

    def _drop_index(self, index_name: str) -> None:
        try:
            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                drop_index_sql = f'DROP INDEX IF EXISTS "{self.schema}"."{index_name}";'
                sess.execute(text(drop_index_sql))
        except Exception as e:
            logger.error(f"Error dropping index '{index_name}': {e}", exc_info=True)
            raise

    def _clean_content(self, content: str) -> str:
        return content.replace("\x00", "\ufffd")

    def _get_document_record(self, doc: VectorDocument, filters: dict[str, Any] | None = None, content_hash: str = "") -> dict[str, Any]:
        cleaned_content = self._clean_content(doc.content)
        record_id = doc.id or content_hash

        meta_data = doc.metadata or {}
        if filters:
            meta_data.update(filters)

        return {
            "id": record_id,
            "name": doc.id,
            "meta_data": meta_data,
            "filters": filters,
            "content": cleaned_content,
            "embedding": doc.vector,
            "usage": {},
            "content_hash": content_hash,
            "content_id": doc.id,
        }

    async def _add_documents_impl(self, documents: list[VectorDocument]) -> bool:
        try:
            if not await self.async_exists():
                await self._initialize_db()

            batch_records = []
            for doc in documents:
                try:
                    batch_records.append(self._get_document_record(doc))
                except Exception as e:
                    logger.error(f"Error processing document '{doc.id}': {e}", exc_info=True)

            if self._Session is None:
                raise RuntimeError("Session not initialized")
            if self.table is None:
                raise RuntimeError("Table not initialized")
            table_obj: Any = self.table
            with self._Session() as sess:
                insert_stmt = postgresql.insert(table_obj)
                sess.execute(insert_stmt, batch_records)
                sess.commit()

            return True

        except Exception as e:
            error_str = str(e).lower()
            if "uniqueviolation" in error_str or "duplicate key" in error_str:
                logger.info("Documents already exist in database (skipping duplicate insertion)")
                return True
            else:
                logger.error(
                    f"Error adding documents to PgVector: {clean_error_message(str(e))}",
                    exc_info=True,
                )
                return False

    def _add_documents_impl_sync(self, documents: list[VectorDocument]) -> bool:
        try:
            if not self.exists():
                self._initialize_db_sync()

            batch_records = []
            for doc in documents:
                try:
                    batch_records.append(self._get_document_record(doc))
                except Exception as e:
                    logger.error(f"Error processing document '{doc.id}': {e}", exc_info=True)

            if self._Session is None:
                raise RuntimeError("Session not initialized")
            if self.table is None:
                raise RuntimeError("Table not initialized")
            table_obj: Any = self.table
            with self._Session() as sess:
                insert_stmt = postgresql.insert(table_obj)
                sess.execute(insert_stmt, batch_records)
                sess.commit()

            return True

        except Exception as e:
            error_str = str(e).lower()
            if "uniqueviolation" in error_str or "duplicate key" in error_str:
                logger.info("Documents already exist in database (skipping duplicate insertion)")
                return True
            else:
                logger.error(
                    f"Error adding documents to PgVector: {clean_error_message(str(e))}",
                    exc_info=True,
                )
                return False

    async def _search_impl(self, query_vector: list[float], top_k: int = 5, filters: dict[str, Any] | None = None) -> list[VectorSearchResult]:
        return await asyncio.to_thread(self._search_impl_sync, query_vector, top_k, filters)

    def _search_impl_sync(self, query_vector: list[float], top_k: int, filters: dict[str, Any] | None = None) -> list[VectorSearchResult]:
        try:
            if not self.exists():
                return []

            if self.table is None:
                raise RuntimeError("Table not initialized")
            columns = [
                self.table.c.id,
                self.table.c.name,
                self.table.c.meta_data,
                self.table.c.content,
                self.table.c.embedding,
            ]

            stmt = select(*columns)

            if filters is not None:
                if self.table is None:
                    raise RuntimeError("Table not initialized")
                stmt = stmt.where(self.table.c.meta_data.contains(filters))

            if self.distance_metric == Distance.L2:
                stmt = stmt.order_by(self.table.c.embedding.l2_distance(query_vector))
            elif self.distance_metric == Distance.COSINE:
                stmt = stmt.order_by(self.table.c.embedding.cosine_distance(query_vector))
            elif self.distance_metric == Distance.INNER_PRODUCT:
                stmt = stmt.order_by(self.table.c.embedding.max_inner_product(query_vector))
            else:
                logger.warning(f"Unknown distance metric: {self.distance_metric}")
                return []

            stmt = stmt.limit(top_k)

            try:
                if self._Session is None:
                    raise RuntimeError("Session not initialized")
                with self._Session() as sess, sess.begin():
                    if self.vector_index is not None:
                        if isinstance(self.vector_index, Ivfflat):
                            sess.execute(text(f"SET LOCAL ivfflat.probes = {self.vector_index.probes}"))
                        elif isinstance(self.vector_index, HNSW):
                            sess.execute(text(f"SET LOCAL hnsw.ef_search = {self.vector_index.ef_search}"))
                    results = sess.execute(stmt).fetchall()
            except Exception as e:
                logger.error(f"Error performing vector search: {e}", exc_info=True)
                return []

            search_results = []
            for result in results:
                similarity: float
                if self.distance_metric == Distance.COSINE:
                    import numpy as np

                    embedding_array = np.array(result.embedding)
                    query_array = np.array(query_vector)

                    dot_product = np.dot(embedding_array, query_array)
                    norm_a = np.linalg.norm(embedding_array)
                    norm_b = np.linalg.norm(query_array)

                    if norm_a == 0 or norm_b == 0:
                        similarity = 0.0
                    else:
                        similarity = float(dot_product / (norm_a * norm_b))

                elif self.distance_metric == Distance.L2:
                    import numpy as np

                    embedding_array = np.array(result.embedding)
                    query_array = np.array(query_vector)

                    l2_distance = np.linalg.norm(embedding_array - query_array)
                    similarity = float(1 / (1 + l2_distance))

                else:
                    import numpy as np

                    embedding_array = np.array(result.embedding)
                    query_array = np.array(query_vector)

                    inner_product = np.dot(embedding_array, query_array)
                    similarity = float(max(0, min(1, (inner_product + 1) / 2)))

                search_results.append(
                    VectorSearchResult(
                        content=result.content,
                        score=similarity,
                        metadata=result.meta_data if result.meta_data else {},
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Error searching PgVector: {clean_error_message(str(e))}", exc_info=True)
            return []

    async def _get_documents_impl(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            if not await self.async_exists():
                return {"ids": [], "contents": [], "metadatas": []}

            if self.table is None:
                raise RuntimeError("Table not initialized")
            columns = [self.table.c.id, self.table.c.content, self.table.c.meta_data]
            stmt = select(*columns)

            if ids:
                if self.table is None:
                    raise RuntimeError("Table not initialized")
                stmt = stmt.where(self.table.c.id.in_(ids))
            if limit:
                stmt = stmt.limit(limit)

            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                results = sess.execute(stmt).fetchall()

                result_ids = []
                contents = []
                metadatas = []

                for result in results:
                    result_ids.append(result.id)
                    contents.append(result.content)
                    metadatas.append(result.meta_data if result.meta_data else {})

                return {"ids": result_ids, "contents": contents, "metadatas": metadatas}

        except Exception as e:
            logger.error(
                f"Error getting documents from PgVector: {clean_error_message(str(e))}",
                exc_info=True,
            )
            return {"ids": [], "contents": [], "metadatas": []}

    def _get_documents_impl_sync(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            if not self.exists():
                return {"ids": [], "contents": [], "metadatas": []}

            if self.table is None:
                raise RuntimeError("Table not initialized")
            columns = [self.table.c.id, self.table.c.content, self.table.c.meta_data]
            stmt = select(*columns)

            if ids:
                if self.table is None:
                    raise RuntimeError("Table not initialized")
                stmt = stmt.where(self.table.c.id.in_(ids))
            if limit:
                stmt = stmt.limit(limit)

            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                results = sess.execute(stmt).fetchall()

                result_ids = []
                contents = []
                metadatas = []

                for result in results:
                    result_ids.append(result.id)
                    contents.append(result.content)
                    metadatas.append(result.meta_data if result.meta_data else {})

                return {"ids": result_ids, "contents": contents, "metadatas": metadatas}

        except Exception as e:
            logger.error(
                f"Error getting documents from PgVector: {clean_error_message(str(e))}",
                exc_info=True,
            )
            return {"ids": [], "contents": [], "metadatas": []}

    async def _count_documents_impl(self) -> int:
        try:
            if not await self.async_exists():
                return 0

            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                if self.table is None:
                    raise RuntimeError("Table not initialized")
                stmt = select(func.count(self.table.c.id))
                result = sess.execute(stmt).scalar()
                return int(result) if result is not None else 0

        except Exception as e:
            logger.error(f"Error counting documents in PgVector: {e}", exc_info=True)
            return 0

    def _count_documents_impl_sync(self) -> int:
        try:
            if not self.exists():
                return 0

            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                if self.table is None:
                    raise RuntimeError("Table not initialized")
                stmt = select(func.count(self.table.c.id))
                result = sess.execute(stmt).scalar()
                return int(result) if result is not None else 0

        except Exception as e:
            logger.error(f"Error counting documents in PgVector: {e}", exc_info=True)
            return 0

    async def async_exists(self) -> bool:
        return await asyncio.to_thread(self.table_exists)

    def exists(self) -> bool:
        return self.table_exists()

    def keyword_search(self, query: str, top_k: int) -> list[VectorSearchResult]:
        try:
            if not self.exists():
                return []

            if self.table is None:
                raise RuntimeError("Table not initialized")

            search_query = func.to_tsquery(self.content_language, query)
            rank_func = func.ts_rank(func.to_tsvector(self.content_language, self.table.c.content), search_query)

            stmt = (
                select(
                    self.table.c.id,
                    self.table.c.content,
                    self.table.c.meta_data,
                    rank_func.label("rank"),
                )
                .where(func.to_tsvector(self.content_language, self.table.c.content).match(search_query))
                .order_by(rank_func.desc())
                .limit(top_k)
            )

            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                results = sess.execute(stmt).fetchall()

            search_results = []
            for result in results:
                search_results.append(
                    VectorSearchResult(
                        content=result.content or "",
                        score=float(result.rank) if result.rank is not None else 0.0,
                        metadata=result.meta_data if result.meta_data else {},
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Error performing keyword search: {e}", exc_info=True)
            return []

    async def _search_text_impl(self, query: str, top_k: int) -> list[VectorSearchResult]:
        return await asyncio.to_thread(self.keyword_search, query, top_k)

    def _search_text_impl_sync(self, query: str, top_k: int) -> list[VectorSearchResult]:
        return self.keyword_search(query, top_k)

    async def _delete_impl(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        return await asyncio.to_thread(self._delete_impl_sync, ids, where)

    def _delete_impl_sync(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> bool:
        try:
            if not self.exists():
                return False

            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                if ids is not None:
                    if self.table is None:
                        raise RuntimeError("Table not initialized")
                    stmt = delete(self.table).where(self.table.c.id.in_(ids))
                elif where is not None:
                    if self.table is None:
                        raise RuntimeError("Table not initialized")
                    stmt = delete(self.table).where(self.table.c.meta_data.contains(where))
                else:
                    return False

                result = sess.execute(stmt)
                return bool(result.rowcount and result.rowcount > 0)

        except Exception as e:
            logger.error(
                f"Error deleting documents from PgVector: {clean_error_message(str(e))}",
                exc_info=True,
            )
            return False

    async def _update_impl(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        return await asyncio.to_thread(self._update_impl_sync, ids, metadatas, documents, embeddings)

    def _update_impl_sync(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        try:
            if not self.exists():
                return False

            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                for i, doc_id in enumerate(ids):
                    update_data = {}

                    if metadatas and i < len(metadatas):
                        update_data["meta_data"] = metadatas[i]

                    if documents and i < len(documents):
                        content_value: Any = documents[i]
                        update_data["content"] = content_value
                        if self.embedder and embeddings is None:
                            new_embedding = self.embedder.embed_sync(documents[i])
                            update_data["embedding"] = new_embedding

                    if embeddings and i < len(embeddings):
                        embedding_value: Any = embeddings[i]
                        update_data["embedding"] = embedding_value

                    if update_data:
                        if self.table is None:
                            raise RuntimeError("Table not initialized")
                        stmt = update(self.table).where(self.table.c.id == doc_id).values(**update_data)
                        sess.execute(stmt)

                return True

        except Exception as e:
            logger.error(
                f"Error updating documents in PgVector: {clean_error_message(str(e))}",
                exc_info=True,
            )
            return False

    async def _upsert_impl(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        return await asyncio.to_thread(self._upsert_impl_sync, ids, metadatas, documents, embeddings)

    def _upsert_impl_sync(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> bool:
        try:
            if not self.exists():
                self._initialize_db_sync()

            if self._Session is None:
                raise RuntimeError("Session not initialized")
            with self._Session() as sess, sess.begin():
                for i, doc_id in enumerate(ids):
                    content = documents[i] if documents and i < len(documents) else ""
                    metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                    embedding = embeddings[i] if embeddings and i < len(embeddings) else None

                    if embedding is None and content and self.embedder:
                        embedding = self.embedder.embed_sync(content)

                    record_data = {
                        "id": doc_id,
                        "name": doc_id,
                        "meta_data": metadata,
                        "content": content,
                        "embedding": embedding,
                        "usage": {},
                        "content_hash": "",
                        "content_id": doc_id,
                    }

                    from sqlalchemy.dialects import postgresql

                    if self.table is None:
                        raise RuntimeError("Table not initialized")
                    insert_stmt = postgresql.insert(self.table).values(record_data)
                    upsert_stmt = insert_stmt.on_conflict_do_update(
                        index_elements=["id"],
                        set_={
                            "name": insert_stmt.excluded.name,
                            "meta_data": insert_stmt.excluded.meta_data,
                            "content": insert_stmt.excluded.content,
                            "embedding": insert_stmt.excluded.embedding,
                            "usage": insert_stmt.excluded.usage,
                            "content_hash": insert_stmt.excluded.content_hash,
                            "content_id": insert_stmt.excluded.content_id,
                        },
                    )
                    sess.execute(upsert_stmt)

                return True

        except Exception as e:
            logger.error(
                f"Error upserting documents in PgVector: {clean_error_message(str(e))}",
                exc_info=True,
            )
            return False
