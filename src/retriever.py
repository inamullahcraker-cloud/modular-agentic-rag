import logging
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.embedding_service import BaseEmbeddingError, EmbeddingService
from src.loader import Document
from src.vectordatabase import StoreError, VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class RetrieverError(Exception):
    """Base exception for all retriever operations."""


class InvalidQueryError(RetrieverError):
    """Raised when top_k is invalid."""


class RetrievalError(RetrieverError):
    """Raised when downstream embedding or vector search fails."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class RetrieverConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_k: int = Field(default=4, gt=0, description="Default number of chunks to retrieve")
    min_score: float = Field(
        default=0.3,
        ge=-1.0,
        le=1.0,
        description="Minimum cosine similarity. Chunks below this are dropped. Tune on real questions.",
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class Retriever:
    """Connects EmbeddingService and VectorStore.

    Query validation happens once, in RAGPipeline. This class trusts its inputs
    (the embedding service still rejects empty queries) and does not log errors:
    it raises, and the caller logs once.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        config: Optional[RetrieverConfig] = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.config = config or RetrieverConfig()

        logger.info(
            "Retriever ready (top_k=%d, min_score=%.2f, %d vectors)",
            self.config.top_k,
            self.config.min_score,
            len(self.vector_store),
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """Returns (Document, score) pairs sorted by score descending.

        Results below `config.min_score` are removed, so an empty list means
        "nothing relevant found", not only "store is empty".
        """
        k = self.config.top_k if top_k is None else top_k
        if k <= 0:
            raise InvalidQueryError(f"top_k must be > 0, got {k}")

        if len(self.vector_store) == 0:
            logger.warning("Retrieval on an empty VectorStore.")
            return []

        try:
            query_vector = self.embedding_service.embed_query(query)
            results = self.vector_store.similarity_search(query_vector, k=k)
        except (BaseEmbeddingError, StoreError) as exc:
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        relevant = [(doc, score) for doc, score in results if score >= self.config.min_score]

        logger.info(
            "Retrieved %d/%d chunk(s) above min_score=%.2f (best=%.3f)",
            len(relevant),
            len(results),
            self.config.min_score,
            results[0][1] if results else float("nan"),
        )
        return relevant