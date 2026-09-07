"""In-memory BM25 sparse index over the same chunks stored in Qdrant.

Rebuilt lazily from a loader callback (the vector store's stored payloads)
rather than maintained as a second persistent store, so there is exactly
one source of truth for chunk content. Call ``invalidate()`` after any
ingest so the next search rebuilds with the new chunks.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Callable

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class _Corpus:
    filenames: list[str]
    texts: list[str]
    bm25: BM25Okapi


class BM25Index:
    def __init__(self, loader: Callable[[], list[dict]]) -> None:
        self._loader = loader
        self._lock = threading.Lock()
        self._corpus: _Corpus | None = None

    def invalidate(self) -> None:
        with self._lock:
            self._corpus = None

    def _ensure_built(self) -> _Corpus | None:
        if self._corpus is not None:
            return self._corpus
        with self._lock:
            if self._corpus is not None:
                return self._corpus
            rows = [r for r in self._loader() if r.get("text")]
            if not rows:
                return None
            tokenized = [_tokenize(r["text"]) for r in rows]
            self._corpus = _Corpus(
                filenames=[r.get("filename") for r in rows],
                texts=[r["text"] for r in rows],
                bm25=BM25Okapi(tokenized),
            )
            return self._corpus

    def search(self, query: str, limit: int = 5) -> list[dict]:
        corpus = self._ensure_built()
        if corpus is None:
            return []
        scores = corpus.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        return [
            {"score": float(scores[i]), "filename": corpus.filenames[i], "text": corpus.texts[i]}
            for i in ranked
            if scores[i] > 0
        ]
