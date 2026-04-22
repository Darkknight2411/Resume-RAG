from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ResumeFile:
    name: str
    content: bytes


@dataclass(slots=True)
class ResumeChunk:
    source_name: str
    chunk_index: int
    text: str


@dataclass(slots=True)
class SearchResult:
    source_name: str
    chunk_index: int
    text: str
    score: float


@dataclass(slots=True)
class IndexSummary:
    source_count: int
    chunk_count: int

