# CloudServe support triage

Capstone project, Forward Deployed AI Engineering. Sanjaya Kumar Panigrahy.

CloudServe's documentation already answers about seven in ten tickets, yet more than half are
escalated. This system answers documented questions from the knowledge base, with citations,
and hands everything else to a person with a summary and the relevant articles attached.
Every decision is logged so it can be explained later.

> Status: week two, day one – environment and project structure (backlog item B-01).

## Requirements

- Python 3.10 or later
- Git
- An OpenRouter API key (free tier)
- About 4 GB of free disk space (embedding model and vector store)

## Setup

```bash
git clone <repository-url> cloudserve-support-triage
cd cloudserve-support-triage
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env                 # then put your key in .env
python scripts/check_model.py        # should print 200 and "ready"
```

## Run

To be completed as components are built:

```bash
# full unattended run over any ticket file
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/
```

## Tests

```bash
python -m pytest tests/ -v
```

## Repository structure

| Path | Contents |
|---|---|
| `src/` | Application code, one module per component |
| `prompts/` | Versioned prompt library (register in `prompts/README.md`) |
| `tests/` | Test suite |
| `evaluation/` | Harness, results and analysis |
| `docs/` | Architecture notes |
| `data/` | Corpus and validation tickets only (see `data/README.md`) |
| `scripts/` | Utility scripts, including the Stage 1 data profile |
| `.github/workflows/` | Continuous integration |

## Attribution

Parts of this repository were drafted with AI assistance (Claude); see the AI tool-use
declaration in the project report. Third-party libraries are listed in `requirements.txt`.
