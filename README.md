# CloudServe support triage

Capstone project, Forward Deployed AI Engineering. Sanjaya Kumar Panigrahy.

CloudServe's help articles already answer about seven in ten support tickets, but more than
half of all tickets are still passed to senior staff. This system:

- answers a ticket from the help articles when it can, and shows which article each answer came from;
- passes every other ticket to a person, with a short summary and the useful articles attached;
- records every decision and the reason for it, so any answer can be explained later.

> Status: week two. Setup (B-01), ticket reading (B-02) and the run harness (B-05) are done.
> The other steps are being built.

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

```bash
# checkpoint: print one normalised ticket per channel
python -m src.ingest --input data/validation_tickets.json

# full unattended run over any ticket file; results go to a dated folder inside --output
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/
```

Right now the system is a skeleton: it sends every ticket to a person and says why.
The real steps (classify, search, decide, answer, check) are added in B-04 to B-09.

## Tests

```bash
python -m pytest tests/ -v
```

## Repository structure

| Path | Contents |
|---|---|
| `src/` | The system's code, one file per step |
| `prompts/` | The prompts, with a version number on each (list in `prompts/README.md`) |
| `tests/` | Automatic tests |
| `evaluation/` | The run harness, its results and the metrics |
| `docs/` | How the system is designed |
| `data/` | Help articles and validation tickets only (see `data/README.md`) |
| `scripts/` | Small helper scripts, including the Stage 1 data analysis |
| `.github/workflows/` | Runs the tests automatically on every push |

## Attribution

Parts of this code were written with help from an AI assistant (Claude). The project report
says where. Other people's libraries are listed in `requirements.txt`.
