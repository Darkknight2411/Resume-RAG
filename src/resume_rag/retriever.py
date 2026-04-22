from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from resume_rag.embedding import embed_query
from resume_rag.schemas import SearchResult
from resume_rag.vector_db import ResumeVectorDB


class ResumeRetriever:
    def __init__(self, vectorizer: TfidfVectorizer, vector_db: ResumeVectorDB) -> None:
        self.vectorizer = vectorizer
        self.vector_db = vector_db

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        query_vector = embed_query(self.vectorizer, query)
        return self.vector_db.search(query_vector=query_vector, top_k=top_k)

