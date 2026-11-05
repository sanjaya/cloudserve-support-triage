"""All settings in one place. Values come from the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STORAGE_DIR = ROOT / "storage"

# Language model: OpenRouter (key from openrouter.ai). Any OpenRouter model name works here;
# names ending in ":free" cost nothing.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct")
# Only for "reasoning" models (they think before answering): set to low / medium / high, and
# leave LLM_REASONING_BUDGET extra tokens for the thinking. Empty for ordinary models.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "")
LLM_REASONING_BUDGET = int(os.getenv("LLM_REASONING_BUDGET", "1024"))
LLM_CACHE_PATH = os.getenv("LLM_CACHE_PATH", str(STORAGE_DIR / "llm_cache.sqlite"))
# Free-tier limits: wait at least this many seconds between calls, and try this many times.
LLM_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S", "2"))
LLM_MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "6"))
# Set to 1 to stop all model calls (the automatic tests do this).
LLM_DISABLED = os.getenv("LLM_DISABLED", "0") == "1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", str(STORAGE_DIR / "chroma"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{STORAGE_DIR / 'decisions.db'}")
# Automatic answers need at least this confidence (FR-04). Chosen 5 Nov from the development-set
# sweep (scripts/sweep_threshold.py): 0.70 to 0.95 all allow only tickets where both methods agree.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
# How articles are split into passages: "section" or "fixed". "section" chosen on 2 Nov (scripts/compare_chunking.py).
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "section")
# Passages less similar than this (0 to 1) are dropped. Chosen on 2 Nov from the
# development-set sweep in scripts/compare_chunking.py (see Stage 5 revision log, change 1).
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.35"))

# Which version of prompt PR-03 writes the answers (B-08): "1.0" or "1.1".
# 1.1 chosen 5 Nov on development tickets (scripts/evaluate_answers.py): it sends 56 of 62
# answerable tickets, against 13 for 1.0. Neither version can tell which tickets the articles
# cannot answer (Stage 5 revision log).
ANSWER_PROMPT_VERSION = os.getenv("ANSWER_PROMPT_VERSION", "1.1")

# Classifier (B-06). The trained model file is committed, so a clean checkout can use it.
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", str(ROOT / "models" / "classifier.joblib"))
# The measured accuracy of each agreement case (written by scripts/evaluate_classifier.py,
# committed). It is the confidence the classifier reports. See src/classify.py.
CLASSIFIER_CALIBRATION_PATH = os.getenv("CLASSIFIER_CALIBRATION_PATH", str(ROOT / "models" / "agreement_calibration.json"))

# The 22 intents and 3 urgency levels in the data (Dataset Guide, ticket schema).
INTENTS = (
    "account_access", "api_key_issue", "api_usage_question", "authentication_failure", "billing_query",
    "compliance_request", "configuration_help", "data_export", "data_residency", "database_issue",
    "deployment_failure", "feature_request", "integration_help", "onboarding", "performance_degradation",
    "quota_or_overage", "rate_limit", "rollback_request", "security_incident", "sso_configuration",
    "unclear_request", "webhook_issue",
)
URGENCIES = ("high", "medium", "low")

# FR-05: these kinds of tickets always go to a person, however sure the system is.
BLOCKED_INTENTS = frozenset({"security_incident", "compliance_request", "feature_request", "unclear_request"})
