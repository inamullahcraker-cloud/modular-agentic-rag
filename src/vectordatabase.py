import hashlib
import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import faiss
import numpy as np

from src.loader import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class StoreError(Exception):
    """Base exception for vector store operations."""


class VectorStoreValidationError(StoreError):
    """Raised when input or dimension checks fail."""


class VectorStorePersistenceError(StoreError):
    """Raised when saving or loading fails."""


class VectorStoreSearchError(StoreError):
    """Raised when a search fails."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def document_id(doc: Document) -> str:
    """Deterministic ID for a chunk: same file + page + chunk position + text => same ID.

    Used to skip duplicates when the same file is ingested twice.
    """
    meta = doc.metadata
    raw = "|".join(
        [
            str(meta.get("file_name", "")),
            str(meta.get("page_number", "")),
            str(meta.get("chunk_index", "")),
            doc.page_content,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
class VectorStore:
    """FAISS IndexFlatIP store. Vectors are L2-normalized here, so scores are
    always cosine similarity regardless of the embedder's settings.

    Thread-safe for add / search / save through one lock (fine for a small API;
    FAISS flat indexes are not safe for concurrent add + search).
    """

    INDEX_FILENAME = "index.faiss"
    DOCS_FILENAME = "documents.json"
    MANIFEST_FILENAME = "manifest.json"
    FORMAT_VERSION = 1

    def __init__(
        self,
        dimension: int,
        embedding_model: Optional[str] = None,
        index: Optional[faiss.IndexFlatIP] = None,
        documents: Optional[List[Document]] = None,
    ) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise VectorStoreValidationError(f"dimension must be a positive integer, got {dimension!r}")

        self.dimension = dimension
        self.embedding_model = embedding_model
        self._index = index if index is not None else faiss.IndexFlatIP(dimension)
        self._documents: List[Document] = documents if documents is not None else []
        self._ids: Set[str] = {document_id(d) for d in self._documents}
        self._lock = threading.RLock()

        logger.info(
            "VectorStore ready (dimension=%d, model=%s, %d vectors)",
            dimension,
            embedding_model or "unspecified",
            self._index.ntotal,
        )

    def __len__(self) -> int:
        return self._index.ntotal

    # -----------------------------------------------------------------------
    # Add / search
    # -----------------------------------------------------------------------
    def filter_new(self, documents: Sequence[Document]) -> List[Document]:
        """Returns only documents not already in the store (and without in-batch duplicates).

        Call this BEFORE embedding to avoid paying for embeddings you will discard.
        """
        with self._lock:
            seen = set(self._ids)
            fresh: List[Document] = []
            for doc in documents:
                doc_id = document_id(doc)
                if doc_id not in seen:
                    seen.add(doc_id)
                    fresh.append(doc)
            return fresh

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Adds documents with their embeddings. Duplicates are skipped.

        Returns:
            Number of documents actually added.
        """
        if len(documents) != len(embeddings):
            raise VectorStoreValidationError(
                f"Length mismatch: {len(documents)} documents vs {len(embeddings)} embeddings."
            )
        if not documents:
            return 0

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise VectorStoreValidationError(
                f"Expected embeddings of shape (n, {self.dimension}), got {vectors.shape}"
            )

        with self._lock:
            keep: List[int] = []
            new_ids: List[str] = []
            seen = set(self._ids)
            for i, doc in enumerate(documents):
                doc_id = document_id(doc)
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                keep.append(i)
                new_ids.append(doc_id)

            skipped = len(documents) - len(keep)
            if skipped:
                logger.info("Skipped %d duplicate chunk(s).", skipped)
            if not keep:
                return 0

            to_add = np.ascontiguousarray(vectors[keep])
            faiss.normalize_L2(to_add)  # in place; makes inner product == cosine

            self._index.add(to_add)
            self._documents.extend(documents[i] for i in keep)
            self._ids.update(new_ids)

            logger.info("Added %d chunk(s). Total: %d", len(keep), self._index.ntotal)
            return len(keep)

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """Returns up to k (Document, cosine_score) pairs, best first."""
        query = np.asarray([query_embedding], dtype=np.float32)
        if query.shape[1] != self.dimension:
            raise VectorStoreValidationError(
                f"Query dimension mismatch: expected {self.dimension}, got {query.shape[1]}"
            )
        if k <= 0:
            raise VectorStoreSearchError(f"k must be positive, got {k}")

        query = np.ascontiguousarray(query)
        faiss.normalize_L2(query)

        with self._lock:
            if self._index.ntotal == 0:
                return []
            scores, indices = self._index.search(query, min(k, self._index.ntotal))
            return [
                (self._documents[idx], float(score))
                for idx, score in zip(indices[0], scores[0])
                if idx != -1
            ]

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------
    def save(self, dir_path: Union[str, Path]) -> None:
        """Saves index, documents and manifest.

        Files are written to a temporary sibling directory first and then swapped in,
        so a crash while writing never leaves a half-written store at `dir_path`.
        """
        target = Path(dir_path)
        tmp = target.with_name(target.name + ".tmp")
        old = target.with_name(target.name + ".old")

        try:
            with self._lock:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir()

                faiss.write_index(self._index, str(tmp / self.INDEX_FILENAME))

                docs_payload = [
                    {"page_content": d.page_content, "metadata": d.metadata} for d in self._documents
                ]
                (tmp / self.DOCS_FILENAME).write_text(
                    json.dumps(docs_payload, ensure_ascii=False), encoding="utf-8"
                )

                manifest = {
                    "format_version": self.FORMAT_VERSION,
                    "embedding_model": self.embedding_model,
                    "dimension": self.dimension,
                    "document_count": len(self._documents),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                (tmp / self.MANIFEST_FILENAME).write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )

            # Swap. If the process dies between the two replaces, the previous
            # store is still available at "<dir>.old".
            if target.exists():
                shutil.rmtree(old, ignore_errors=True)
                os.replace(target, old)
            os.replace(tmp, target)
            shutil.rmtree(old, ignore_errors=True)

            logger.info("Saved VectorStore to '%s' (%d vectors)", target, len(self))

        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise VectorStorePersistenceError(f"Failed to save VectorStore to '{target}': {exc}") from exc

    @classmethod
    def load(
        cls,
        dir_path: Union[str, Path],
        expected_embedding_model: Optional[str] = None,
    ) -> "VectorStore":
        """Loads a store saved by `save`.

        Args:
            dir_path: Directory containing the store files.
            expected_embedding_model: If given, loading fails when the store was built
                with a different model. Vectors from different models are not comparable
                even when the dimension matches.
        """
        source = Path(dir_path)
        required = [cls.INDEX_FILENAME, cls.DOCS_FILENAME, cls.MANIFEST_FILENAME]
        missing = [name for name in required if not (source / name).exists()]
        if missing:
            raise VectorStorePersistenceError(f"Missing files in '{source}': {', '.join(missing)}")

        try:
            manifest: Dict[str, Any] = json.loads((source / cls.MANIFEST_FILENAME).read_text(encoding="utf-8"))
            raw_docs: List[Dict[str, Any]] = json.loads((source / cls.DOCS_FILENAME).read_text(encoding="utf-8"))
            index = faiss.read_index(str(source / cls.INDEX_FILENAME))
        except Exception as exc:
            raise VectorStorePersistenceError(f"Failed to read VectorStore from '{source}': {exc}") from exc

        stored_model = manifest.get("embedding_model")
        if expected_embedding_model and stored_model and stored_model != expected_embedding_model:
            raise VectorStorePersistenceError(
                f"Embedding model mismatch: store built with '{stored_model}', "
                f"but '{expected_embedding_model}' was requested. Re-ingest your documents."
            )

        documents = [
            Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
            for item in raw_docs
        ]

        if index.ntotal != len(documents):
            raise VectorStorePersistenceError(
                f"Index size ({index.ntotal}) does not match document count ({len(documents)})."
            )
        if manifest.get("dimension") not in (None, index.d):
            raise VectorStorePersistenceError(
                f"Manifest dimension ({manifest['dimension']}) does not match index dimension ({index.d})."
            )

        return cls(
            dimension=index.d,
            embedding_model=stored_model,
            index=index,
            documents=documents,
        )