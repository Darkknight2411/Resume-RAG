from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from resume_rag.schemas import IndexSummary, ResumeChunk, SearchResult


class ResumeVectorDB:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS resume_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    vector_size INTEGER NOT NULL
                )
                """
            )
            connection.commit()

    def replace_chunks(self, chunks: list[ResumeChunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk count and vector count must match.")

        self.initialize()
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("DELETE FROM resume_chunks")
            connection.executemany(
                """
                INSERT INTO resume_chunks (source_name, chunk_index, text, vector, vector_size)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.source_name,
                        chunk.chunk_index,
                        chunk.text,
                        _serialize_vector(vector),
                        int(vector.shape[0]),
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            connection.commit()

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        records = self._fetch_records()
        if not records:
            return []

        matrix = np.vstack([record["vector"] for record in records]).astype(np.float32)
        normalized_matrix = _normalize_rows(matrix)

        normalized_query = _normalize_vector(query_vector)
        if normalized_query is None:
            return []

        scores = normalized_matrix @ normalized_query
        ranked_indexes = np.argsort(scores)[::-1][:top_k]

        results: list[SearchResult] = []
        for index in ranked_indexes:
            record = records[index]
            results.append(
                SearchResult(
                    source_name=record["source_name"],
                    chunk_index=record["chunk_index"],
                    text=record["text"],
                    score=float(scores[index]),
                )
            )
        return results

    def get_summary(self) -> IndexSummary | None:
        if not self.database_path.exists():
            return None

        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT source_name)
                FROM resume_chunks
                """
            ).fetchone()

        if row is None or row[0] == 0:
            return None

        chunk_count, source_count = row
        return IndexSummary(source_count=source_count, chunk_count=chunk_count)

    def _fetch_records(self) -> list[dict[str, object]]:
        self.initialize()
        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT source_name, chunk_index, text, vector, vector_size
                FROM resume_chunks
                ORDER BY source_name, chunk_index
                """
            ).fetchall()

        records: list[dict[str, object]] = []
        for source_name, chunk_index, text, vector_blob, vector_size in rows:
            records.append(
                {
                    "source_name": source_name,
                    "chunk_index": chunk_index,
                    "text": text,
                    "vector": _deserialize_vector(vector_blob, vector_size),
                }
            )
        return records


def _serialize_vector(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _deserialize_vector(blob: bytes, vector_size: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=vector_size).copy()


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return matrix / safe_norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None
    return vector.astype(np.float32) / norm

