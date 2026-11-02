"""All settings in one place. Values come from the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STORAGE_DIR = ROOT / "storage"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", str(STORAGE_DIR / "chroma"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{STORAGE_DIR / 'decisions.db'}")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
# How articles are split into passages: "section" or "fixed". "section" chosen on 2 Nov (scripts/compare_chunking.py).
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "section")
# Passages less similar than this (0 to 1) are dropped. Chosen on 2 Nov from the
# development-set sweep in scripts/compare_chunking.py (see Stage 5 revision log, change 1).
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.35"))

# FR-05: these kinds of tickets always go to a person, however sure the system is.
BLOCKED_INTENTS = frozenset({"security_incident", "compliance_request", "feature_request", "unclear_request"})
