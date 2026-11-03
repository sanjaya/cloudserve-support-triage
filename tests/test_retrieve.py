"""Tests for searching the help articles (FR-03). Test ids T-03a, T-03b and T-03c come
from the Stage 3 specification. They use the offline stand-in embedder (see conftest.py),
so they check the plumbing, not the quality of the real model."""
import json

import pytest

from src import config
from src.retrieve import HashingEmbedder, Retriever, load_articles, split_by_section, split_fixed

ARTICLES = load_articles()
DOC_IDS = {a["doc_id"] for a in ARTICLES}


@pytest.fixture(scope="module")
def retriever(tmp_path_factory):
    return Retriever(strategy="section", embedder=HashingEmbedder(),
                     store_path=str(tmp_path_factory.mktemp("chroma")), min_score=0.0)


def test_section_split_keeps_resolution_steps_together():
    rollback = next(a for a in ARTICLES if "Rolling back" in a["title"])
    passages = split_by_section(rollback)
    sections = [p.section for p in passages]
    assert sections == ["symptoms", "common-causes", "resolution", "notes"]
    resolution = passages[sections.index("resolution")].text
    assert "1. " in resolution and "4. " in resolution          # all four steps in one passage
    assert all(p.text.startswith(rollback["title"]) for p in passages)


def test_fixed_split_covers_the_whole_article():
    article = ARTICLES[0]
    parts = split_fixed(article, size=200, overlap=50)
    assert parts[0].text.startswith(article["content"][:50])
    assert article["content"].rstrip().endswith(parts[-1].text.rstrip()[-30:])


def test_every_article_is_stored(retriever):
    stored = {m["doc_id"] for m in retriever.collection.get()["metadatas"]}
    assert stored == DOC_IDS


def test_T03a_every_result_points_to_a_real_passage(retriever):
    results = retriever.search("my deployment fails at the health check stage")
    assert results
    for p in results:
        assert p.doc_id in DOC_IDS
        assert retriever.get(p.passage_id) is not None       # a citation could be checked
        assert 0.0 <= p.score <= 1.0


def test_results_are_best_first(retriever):
    scores = [p.score for p in retriever.search("how do I roll back a release to an earlier revision")]
    assert scores == sorted(scores, reverse=True)


def test_T03b_known_question_finds_the_right_article(retriever):
    # Uses words from the rollback article, so even the word-matching stand-in finds it.
    top = retriever.search("roll back to the last healthy revision after a failed release", top_k=3)
    assert any("Rolling back" in p.title for p in top)


def test_T03c_off_topic_ticket_returns_nothing_above_the_minimum(retriever):
    assert retriever.search("zzqx blorf wibble", min_score=0.30) == []


def test_empty_text_returns_nothing(retriever):
    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_store_is_rebuilt_when_articles_change(tmp_path):
    changed = ARTICLES[:3]
    path = tmp_path / "docs.json"
    path.write_text(json.dumps(changed))
    store = str(tmp_path / "chroma")
    first = Retriever(strategy="section", embedder=HashingEmbedder(), store_path=store, articles_path=config.DATA_DIR / "documentation.json")
    assert {m["doc_id"] for m in first.collection.get()["metadatas"]} == DOC_IDS
    second = Retriever(strategy="section", embedder=HashingEmbedder(), store_path=store, articles_path=path)
    assert {m["doc_id"] for m in second.collection.get()["metadatas"]} == {a["doc_id"] for a in changed}


def test_unknown_split_method_is_refused(tmp_path):
    with pytest.raises(ValueError):
        Retriever(strategy="paragraph", embedder=HashingEmbedder(), store_path=str(tmp_path))


def test_search_finds_the_true_closest_passages(retriever):
    """Chroma's search is approximate; with INDEX_SETTINGS it must match checking every passage."""
    import numpy as np
    emb = HashingEmbedder()
    passages = [p for a in ARTICLES for p in split_by_section(a)]
    matrix = np.array(emb.embed_documents([p.text for p in passages]))
    tickets = json.loads((config.DATA_DIR / "validation_tickets.json").read_text())[:30]
    for t in tickets:
        text = f"{t.get('subject') or ''}\n{t.get('body') or ''}".strip()
        scores = matrix @ np.array(emb.embed_query(text))
        exact = {passages[i].passage_id for i in np.argsort(-scores)[:10]}
        found = {p.passage_id for p in retriever.search(text, top_k=10)}
        assert len(exact & found) >= 9          # ties at the cut-off may swap one passage


def test_store_uses_langchain_chroma(retriever):
    from langchain_chroma import Chroma
    assert isinstance(retriever.store, Chroma)
    assert retriever.collection.metadata["hnsw:space"] == "cosine"
