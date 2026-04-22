from resume_rag.chunking import chunk_text


def test_chunk_text_preserves_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(1, 21))

    chunks = chunk_text(text=text, chunk_size=6, chunk_overlap=2)

    assert chunks[0].split() == ["word1", "word2", "word3", "word4", "word5", "word6"]
    assert chunks[1].split()[:2] == ["word5", "word6"]
    assert chunks[-1].split()[-1] == "word20"

