from __future__ import annotations

import sys
from uuid import uuid4
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from resume_rag.config import Settings
from resume_rag.pipeline import ResumeRAGPipeline
from resume_rag.schemas import ResumeFile


def get_session_settings() -> Settings:
    if "rag_session_id" not in st.session_state:
        st.session_state["rag_session_id"] = uuid4().hex

    settings = Settings.from_project_root(PROJECT_ROOT)
    session_artifacts_dir = PROJECT_ROOT / "artifacts" / "sessions" / st.session_state["rag_session_id"]
    settings.database_path = session_artifacts_dir / "resume_vectors.db"
    settings.vectorizer_path = session_artifacts_dir / "tfidf_vectorizer.joblib"
    return settings


def main() -> None:
    st.set_page_config(page_title="RAG Resume Query", page_icon="R", layout="wide")

    settings = get_session_settings()
    pipeline = ResumeRAGPipeline(settings)

    st.title("RAG Resume Query")
    st.caption(
        "Upload resumes, build a local vector database, and retrieve the most relevant resume chunks for a query."
    )

    with st.expander("What is happening in this app?", expanded=True):
        st.markdown(
            """
            1. We parse each resume into raw text.
            2. We split that text into overlapping chunks.
            3. We convert each chunk into a TF-IDF vector embedding.
            4. We save those vectors and their source text in a session-specific SQLite vector DB.
            5. Your query is transformed into the same vector space.
            6. We compare the query against stored vectors and return the top-k results.
            """
        )

    existing_summary = pipeline.get_index_summary()
    if existing_summary is not None:
        st.info(
            f"Existing local vector DB found with {existing_summary.source_count} resumes and "
            f"{existing_summary.chunk_count} chunks."
        )

    st.subheader("1. Upload and Index Resumes")
    uploaded_files = st.file_uploader(
        "Upload resume files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="The app extracts text, chunks it, embeds it, and stores those vectors locally.",
    )

    build_col, info_col = st.columns([1, 1])

    with build_col:
        if st.button("Build / Rebuild Vector DB", type="primary", disabled=not uploaded_files):
            resume_files = [
                ResumeFile(name=uploaded_file.name, content=uploaded_file.getvalue())
                for uploaded_file in uploaded_files
            ]
            with st.spinner("Parsing resumes, chunking text, creating embeddings, and storing vectors..."):
                try:
                    summary = pipeline.build_index(resume_files)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.success(
                        f"Indexed {summary.source_count} resumes into {summary.chunk_count} chunks."
                    )

    with info_col:
        st.markdown(
            """
            **Why chunking matters**

            Resumes are usually too long to treat as one big text block. Chunking breaks them into smaller,
            overlapping pieces so retrieval can focus on the most relevant section.
            """
        )

    st.subheader("2. Query the Retriever")

    query = st.text_input(
        "Ask a resume question",
        placeholder="Which candidate has the strongest backend and SQL experience?",
    )
    top_k = st.slider("Top K matches", min_value=1, max_value=10, value=settings.top_k)

    if st.button("Run Retrieval", disabled=not query):
        with st.spinner("Turning the query into a vector and comparing it with the vector DB..."):
            try:
                results = pipeline.query(query=query, top_k=top_k)
            except (FileNotFoundError, ValueError) as error:
                st.error(str(error))
            else:
                if not results:
                    st.warning("No relevant matches were found for this query.")
                else:
                    best_match = results[0]
                    st.success(
                        f"Top match: {best_match.source_name} (chunk {best_match.chunk_index}, "
                        f"score {best_match.score:.3f})"
                    )

                    for rank, result in enumerate(results, start=1):
                        with st.container(border=True):
                            st.markdown(
                                f"**Rank {rank}: {result.source_name}**  \n"
                                f"Chunk `{result.chunk_index}`  \n"
                                f"Similarity score `{result.score:.3f}`"
                            )
                            st.write(result.text)


if __name__ == "__main__":
    main()
