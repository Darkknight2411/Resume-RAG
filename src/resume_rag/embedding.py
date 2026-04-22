from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from resume_rag.schemas import ResumeChunk


def build_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        norm="l2",
    )


def embed_chunks(
    chunks: list[ResumeChunk],
    max_features: int,
) -> tuple[TfidfVectorizer, np.ndarray]:
    if not chunks:
        raise ValueError("At least one chunk is required to build embeddings.")

    vectorizer = build_vectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(chunk.text for chunk in chunks)
    return vectorizer, matrix.toarray().astype(np.float32)


def embed_query(vectorizer: TfidfVectorizer, query: str) -> np.ndarray:
    matrix = vectorizer.transform([query])
    return matrix.toarray().astype(np.float32)[0]

