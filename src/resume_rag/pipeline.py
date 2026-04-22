from __future__ import annotations

from pathlib import Path

import joblib

from resume_rag.chunking import create_resume_chunks
from resume_rag.config import Settings
from resume_rag.embedding import embed_chunks
from resume_rag.parsers import extract_text
from resume_rag.retriever import ResumeRetriever
from resume_rag.schemas import IndexSummary, ResumeFile, SearchResult
from resume_rag.vector_db import ResumeVectorDB


class ResumeRAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vector_db = ResumeVectorDB(settings.database_path)

    def build_index(self, resume_files: list[ResumeFile]) -> IndexSummary:
        if not resume_files:
            raise ValueError("Upload at least one resume before building the index.")

        chunks = []
        seen_sources: set[str] = set()

        for resume_file in resume_files:
            text = extract_text(resume_file)
            if not text:
                continue

            source_chunks = create_resume_chunks(
                source_name=resume_file.name,
                text=text,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            if not source_chunks:
                continue

            chunks.extend(source_chunks)
            seen_sources.add(resume_file.name)

        if not chunks:
            raise ValueError("No readable text could be extracted from the uploaded resumes.")

        vectorizer, vectors = embed_chunks(chunks=chunks, max_features=self.settings.max_features)
        self.settings.vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, self.settings.vectorizer_path)
        self.vector_db.replace_chunks(chunks=chunks, vectors=vectors)

        return IndexSummary(source_count=len(seen_sources), chunk_count=len(chunks))

    def query(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Enter a question before running retrieval.")

        retriever = self.load_retriever()
        requested_top_k = top_k or self.settings.top_k
        return retriever.retrieve(query=query, top_k=requested_top_k)

    def get_index_summary(self) -> IndexSummary | None:
        return self.vector_db.get_summary()

    def load_retriever(self) -> ResumeRetriever:
        if not self.settings.vectorizer_path.exists():
            raise FileNotFoundError(
                "Vectorizer file not found. Build the vector DB before running a query."
            )

        if not self.settings.database_path.exists():
            raise FileNotFoundError(
                "Vector DB not found. Build the vector DB before running a query."
            )

        vectorizer = joblib.load(self.settings.vectorizer_path)
        return ResumeRetriever(vectorizer=vectorizer, vector_db=self.vector_db)

