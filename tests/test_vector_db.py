from pathlib import Path

import numpy as np

from resume_rag.schemas import ResumeChunk
from resume_rag.vector_db import ResumeVectorDB


def test_vector_db_returns_most_similar_chunk(tmp_path: Path) -> None:
    database = ResumeVectorDB(tmp_path / "resume_vectors.db")
    chunks = [
        ResumeChunk(source_name="alice_resume.pdf", chunk_index=1, text="Python backend APIs"),
        ResumeChunk(source_name="bob_resume.pdf", chunk_index=1, text="Excel reporting finance"),
        ResumeChunk(source_name="carol_resume.pdf", chunk_index=1, text="ML models and Python"),
    ]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.0, 0.2],
        ],
        dtype=np.float32,
    )

    database.replace_chunks(chunks=chunks, vectors=vectors)
    results = database.search(query_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=2)

    assert len(results) == 2
    assert results[0].source_name == "alice_resume.pdf"
    assert results[1].source_name == "carol_resume.pdf"
