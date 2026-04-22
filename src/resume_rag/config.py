from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    project_root: Path
    database_path: Path
    vectorizer_path: Path
    chunk_size: int = 180
    chunk_overlap: int = 40
    top_k: int = 5
    max_features: int = 5000

    @classmethod
    def from_project_root(cls, project_root: Path) -> "Settings":
        artifacts_dir = project_root / "artifacts"
        return cls(
            project_root=project_root,
            database_path=artifacts_dir / "resume_vectors.db",
            vectorizer_path=artifacts_dir / "tfidf_vectorizer.joblib",
        )

