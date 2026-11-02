"""Settings for all automatic tests.

The tests use a small offline stand-in for the embedding model, so they run quickly,
without internet and without downloading the real model (on your Mac or in CI).
The search store goes in a temporary folder, so tests never touch storage/.
"""
import os
import tempfile

os.environ["EMBEDDING_MODEL"] = "hashing"
os.environ["CHROMA_PATH"] = tempfile.mkdtemp(prefix="chroma_test_")
os.environ.setdefault("RETRIEVAL_MIN_SCORE", "0.30")
