"""
memory/semantic_cache.py
Message-level semantic cache — keys on the USER'S MESSAGE only,
not the full system prompt. This avoids the problem where dynamic
fields (turn_count, memory_context) break LangChain's built-in cache.

Flow:
  1. User sends a message
  2. pre-graph: embed message -> search cache
  3a. Cache HIT  -> return cached answer, skip all LLM calls
  3b. Cache MISS -> run graph normally -> save answer to cache
"""

import logging
import os
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.collection import Collection

from ..config import MONGO_CONFIG

logger = logging.getLogger(__name__)


class MessageSemanticCache:
    """
    Caches LLM responses keyed by the semantic meaning of the user's message.
    Stored in MongoDB — works with both local and Atlas (uses cosine similarity
    via Atlas Vector Search when available, falls back to exact match locally).
    """

    def __init__(
        self, client: MongoClient | None = None, score_threshold: float = 1
    ) -> None:
        _client = client or MongoClient(MONGO_CONFIG.uri)
        db = _client[MONGO_CONFIG.db_name]
        self.collection: Collection = db["llm_semantic_cache"]
        self.threshold = score_threshold
        self.embedder = None  # lazy loaded
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.collection.create_index("created_at")

    def _get_embedder(self):
        if self.embedder is None:
            from ..utils.embeddings import EmbeddingHelper
            self.embedder = EmbeddingHelper()
        return self.embedder

    # ── Write ──────────────────────────────────────────────────────────────────

    def save(self, user_message: str, answer: str) -> None:
        """Save a user message + answer pair to the cache."""
        if not os.getenv("OPENAI_API_KEY"):
            return

        try:
            embedding = self._get_embedder().embed(user_message)
            self.collection.insert_one(
                {
                    "user_message": user_message,
                    "answer": answer,
                    "embedding": embedding,
                    "created_at": datetime.now(timezone.utc),
                }
            )
            logger.debug("Cache saved: '%s'", user_message[:60])
        except Exception as e:
            logger.warning("Cache save failed (non-critical): %s", e)

    # ── Read ───────────────────────────────────────────────────────────────────

    def lookup(self, user_message: str) -> str | None:
        """Look up a user message in the cache.

        1. Exact match first — instant, no index latency
        2. Vector search    — semantic similarity for rephrased questions
        """
        if not os.getenv("OPENAI_API_KEY"):
            return None

        exact = self._exact_lookup(user_message)
        if exact:
            return exact

        try:
            query_vec = self._get_embedder().embed(user_message)
            return self._vector_lookup(query_vec, user_message)
        except Exception as e:
            if "SearchNotEnabled" in str(e) or "vectorSearch" in str(e):
                return None
            logger.warning("Cache lookup error (non-critical): %s", e)
            return None

    def _vector_lookup(self, query_vec: list[float], user_message: str) -> str | None:
        """Atlas Vector Search lookup."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "cache_vector_index",
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": 10,
                    "limit": 1,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "answer": 1,
                    "user_message": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        results = list(self.collection.aggregate(pipeline))
        if results and results[0].get("score", 0) >= self.threshold:
            hit = results[0]
            logger.debug("Cache HIT (score=%.3f): '%s'", hit["score"], hit["user_message"][:60])
            return hit["answer"]
        logger.debug("Cache MISS for: '%s'", user_message[:60])
        return None

    def _exact_lookup(self, user_message: str) -> str | None:
        """Fallback: exact string match for local MongoDB."""
        doc = self.collection.find_one(
            {"user_message": user_message}, {"_id": 0, "answer": 1}
        )
        if doc:
            logger.debug("Cache HIT (exact match): '%s'", user_message[:60])
            return doc["answer"]
        return None

    # ── Utils ──────────────────────────────────────────────────────────────────

    def clear(self) -> int:
        result = self.collection.delete_many({})
        logger.info("Cache cleared %d entries.", result.deleted_count)
        return result.deleted_count

    def stats(self) -> dict:
        return {"total_cached": self.collection.count_documents({})}
