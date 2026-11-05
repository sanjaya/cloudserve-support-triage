"""Settings for all automatic tests.

The tests use a small offline stand-in for the embedding model, so they run quickly,
without internet and without downloading the real model (on your Mac or in CI).
The search store and the model-answer cache go in temporary folders, so tests never
touch storage/. Real language-model calls are switched off.
"""
import os
import tempfile

os.environ["EMBEDDING_MODEL"] = "hashing"
os.environ["CHROMA_PATH"] = tempfile.mkdtemp(prefix="chroma_test_")
os.environ.setdefault("RETRIEVAL_MIN_SCORE", "0.30")
# No real language-model calls in tests, and a temporary answer cache.
os.environ["LLM_DISABLED"] = "1"
os.environ["LLM_CACHE_PATH"] = os.path.join(tempfile.mkdtemp(prefix="llm_cache_test_"), "cache.sqlite")
# Tests never use the committed model file (it was trained with the real embedding model).
os.environ["CLASSIFIER_MODEL_PATH"] = os.path.join(tempfile.mkdtemp(prefix="clf_test_"), "none.joblib")
os.environ["CLASSIFIER_CALIBRATION_PATH"] = os.path.join(tempfile.mkdtemp(prefix="cal_test_"), "none.json")
# The decision log goes to a temporary database, never storage/decisions.db.
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(prefix="decisions_test_"), "decisions.db")
