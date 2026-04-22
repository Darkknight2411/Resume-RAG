from __future__ import annotations

from io import BytesIO
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from resume_rag.schemas import ResumeFile


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def extract_text(resume_file: ResumeFile) -> str:
    extension = Path(resume_file.name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type for {resume_file.name}. "
            "Use PDF, DOCX, TXT, or MD."
        )

    if extension == ".pdf":
        return _extract_pdf_text(resume_file.content)
    if extension == ".docx":
        return _extract_docx_text(resume_file.content)
    return _extract_plain_text(resume_file.content)


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return _clean_text("\n".join(pages))


def _extract_docx_text(file_bytes: bytes) -> str:
    document = Document(BytesIO(file_bytes))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return _clean_text("\n".join(paragraphs))


def _extract_plain_text(file_bytes: bytes) -> str:
    return _clean_text(file_bytes.decode("utf-8", errors="ignore"))


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines).strip()

