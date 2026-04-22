from __future__ import annotations

from resume_rag.schemas import ResumeChunk


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    return chunks


def create_resume_chunks(
    source_name: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ResumeChunk]:
    chunk_texts = chunk_text(text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        ResumeChunk(source_name=source_name, chunk_index=index, text=chunk)
        for index, chunk in enumerate(chunk_texts, start=1)
    ]

