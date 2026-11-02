"""Search the help articles (FR-03). Backlog items B-03 (split and store) and B-04 (search).

How it works:
1. Each of the 29 help articles is split into passages.
   - "section" (default): one passage per section of the article
     (symptoms, common causes, resolution, notes). The numbered steps stay together.
   - "fixed": pieces of about 800 characters that overlap a little (the Setup Guide's example).
   scripts/compare_chunking.py measures which one finds the right article more often.
2. Each passage is turned into numbers (an "embedding") and saved in Chroma, a small
   database that runs inside this program and keeps its files in storage/.
3. To search, the ticket text is turned into numbers the same way, and Chroma returns
   the passages that are closest in meaning.
4. Passages that are not similar enough (score below RETRIEVAL_MIN_SCORE) are dropped.
   Returning nothing is better than returning something that does not fit.

The store is built automatically the first time, and rebuilt if the articles,
the splitting method or the embedding model change. storage/ is never committed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from src import config

# ---------------------------------------------------------------- passages


@dataclass
class Passage:
    passage_id: str      # e.g. "DOC-DEPLOY-002#resolution" – what a citation points to
    doc_id: str          # the article it came from
    title: str
    section: str
    text: str
    score: float = 0.0   # similarity to the ticket, 0 to 1 (filled in by search)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def split_by_section(doc: dict[str, Any]) -> list[Passage]:
    """One passage per '## ' section. The article title is added to the top of each
    passage, so a passage still makes sense when read on its own."""
    title = doc["title"]
    parts = re.split(r"^## +", doc["content"], flags=re.M)
    passages = []
    for part in parts[1:]:                    # parts[0] is the title and 'Applies to' line
        heading, _, body = part.partition("\n")
        body = body.strip()
        if not body:
            continue
        section = _slug(heading)
        text = f"{title}\nApplies to: {doc.get('applies_to', '')}\n{heading.strip()}\n{body}"
        passages.append(Passage(f"{doc['doc_id']}#{section}", doc["doc_id"], title, section, text))
    if not passages:                          # an article with no sections: keep it whole
        passages.append(Passage(f"{doc['doc_id']}#all", doc["doc_id"], title, "all", doc["content"]))
    return passages


def split_fixed(doc: dict[str, Any], size: int = 800, overlap: int = 120) -> list[Passage]:
    """Pieces of about `size` characters, each overlapping the last by `overlap`."""
    content = doc["content"]
    passages, start, n = [], 0, 0
    while start < len(content):
        piece = content[start:start + size]
        passages.append(Passage(f"{doc['doc_id']}#part{n}", doc["doc_id"], doc["title"], f"part{n}", piece))
        if start + size >= len(content):
            break
        start += size - overlap
        n += 1
    return passages


SPLITTERS = {"section": split_by_section, "fixed": split_fixed}


def load_articles(path: Path | None = None) -> list[dict[str, Any]]:
    return json.loads((path or config.DATA_DIR / "documentation.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- embeddings


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """The real model, all-MiniLM-L6-v2. Downloaded (about 90 MB) on first use, then cached."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # imported here: slow to load
        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


class HashingEmbedder:
    """A tiny offline stand-in used ONLY by the automatic tests, so they run without
    downloading the real model. It matches words, not meaning. Never use it for results."""

    name = "hashing"

    def __init__(self, dims: int = 512):
        self.dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            v = [0.0] * self.dims
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return vectors


def get_embedder(name: str | None = None) -> Embedder:
    name = name or config.EMBEDDING_MODEL
    return HashingEmbedder() if name == "hashing" else SentenceTransformerEmbedder(name)


# ---------------------------------------------------------------- store and search


class Retriever:
    def __init__(self, strategy: str | None = None, embedder: Embedder | None = None,
                 store_path: str | None = None, min_score: float | None = None,
                 top_k: int | None = None, articles_path: Path | None = None):
        import chromadb
        from chromadb.config import Settings

        self.strategy = strategy or config.RETRIEVAL_STRATEGY
        if self.strategy not in SPLITTERS:
            raise ValueError(f"Unknown splitting method {self.strategy!r}; use one of {list(SPLITTERS)}")
        self.embedder = embedder or get_embedder()
        self.min_score = config.RETRIEVAL_MIN_SCORE if min_score is None else min_score
        self.top_k = top_k or config.RETRIEVAL_TOP_K
        self.articles = load_articles(articles_path)

        client = chromadb.PersistentClient(path=store_path or config.CHROMA_PATH,
                                           settings=Settings(anonymized_telemetry=False))
        name = f"docs_{self.strategy}_{_slug(self.embedder.name)}"[:60]
        fingerprint = self._fingerprint()
        self.collection = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        if (self.collection.metadata or {}).get("fingerprint") != fingerprint or self.collection.count() == 0:
            client.delete_collection(name)
            self.collection = client.create_collection(
                name, metadata={"hnsw:space": "cosine", "fingerprint": fingerprint})
            self._build()

    def _fingerprint(self) -> str:
        raw = json.dumps(self.articles, sort_keys=True) + self.strategy + self.embedder.name
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _build(self) -> None:
        passages = [p for doc in self.articles for p in SPLITTERS[self.strategy](doc)]
        self.collection.add(
            ids=[p.passage_id for p in passages],
            documents=[p.text for p in passages],
            embeddings=self.embedder.embed([p.text for p in passages]),
            metadatas=[{"doc_id": p.doc_id, "title": p.title, "section": p.section} for p in passages],
        )

    @property
    def passage_count(self) -> int:
        return self.collection.count()

    def search(self, text: str, top_k: int | None = None, min_score: float | None = None) -> list[Passage]:
        """Return the best passages for `text`, best first. Passages scoring below the
        minimum are dropped, so the list can be empty. Empty text gives an empty list."""
        if not text or not text.strip():
            return []
        k = min(top_k or self.top_k, self.passage_count)
        threshold = self.min_score if min_score is None else min_score
        found = self.collection.query(query_embeddings=self.embedder.embed([text]), n_results=k)
        passages = []
        for pid, doc, meta, dist in zip(found["ids"][0], found["documents"][0],
                                        found["metadatas"][0], found["distances"][0]):
            score = round(1.0 - dist, 4)       # cosine distance -> similarity
            if score >= threshold:
                passages.append(Passage(pid, meta["doc_id"], meta["title"], meta["section"], doc, score))
        return passages

    def get(self, passage_id: str) -> Passage | None:
        """Look up one passage by id – used to check that a citation points to a real passage."""
        got = self.collection.get(ids=[passage_id])
        if not got["ids"]:
            return None
        meta = got["metadatas"][0]
        return Passage(passage_id, meta["doc_id"], meta["title"], meta["section"], got["documents"][0])
