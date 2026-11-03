# CloudServe support triage

Capstone project, Forward Deployed AI Engineering. Sanjaya Kumar Panigrahy.

CloudServe's help articles already answer about seven in ten support tickets, but more than
half of all tickets are still passed to senior staff. This system:

- answers a ticket from the help articles when it can, and shows which article each answer came from;
- passes every other ticket to a person, with a short summary and the useful articles attached;
- records every decision and the reason for it, so any answer can be explained later.

> Status: week two. Setup (B-01), ticket reading (B-02), the run harness (B-05), searching
> the help articles (B-03, B-04) and classifying tickets (B-06) are done. The other steps are being built.

## Requirements

- Python 3.10 or later
- Git
- An OpenRouter API key (from openrouter.ai) for the language model
- About 4 GB of free disk space (embedding model and vector store)

## Setup

```bash
git clone <repository-url> cloudserve-support-triage
cd cloudserve-support-triage
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env                 # then put your key in .env (OPENROUTER_API_KEY)
python scripts/check_model.py        # should print: openrouter / meta-llama/llama-3.1-8b-instruct: 'ready'
```

## Run

```bash
# checkpoint: print one normalised ticket per channel
python -m src.ingest --input data/validation_tickets.json

# compare the two ways of splitting articles and get the minimum-score table
# (needs data/development_tickets.json, copied from Capstone_Pack/05_Datasets/)
python scripts/compare_chunking.py

# measure the classifier fairly and save the measured confidence (models/agreement_calibration.json)
python scripts/evaluate_classifier.py                   # trained classifier only, no API calls
python scripts/evaluate_classifier.py --llm-sample 160  # full evaluation, asks the language model (OpenRouter)

# retrain the classifier (only needed if the data or the embedding model changes)
python scripts/train_classifier.py

# full unattended run over any ticket file; results go to a dated folder inside --output
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/
```

The first run downloads the embedding model (about 90 MB) and builds the search store in
`storage/`. This happens automatically and takes a minute; later runs reuse both.

Right now the system classifies each ticket and searches the help articles, but still sends
every ticket to a person, with the reason, the labels and the passages it found. Decide,
answer and check come in B-07 to B-09.

How tickets are classified: the language model decides what the ticket is about, and a small
trained classifier (`models/classifier.joblib`, committed) checks that answer. The confidence is
how often that case (both agree / partly agree / disagree) was actually right on the development
tickets, saved in `models/agreement_calibration.json` (committed). If the language model is
unavailable, the trained classifier's answer is used with its own measured confidence.
See `src/classify.py`.

## Tests

```bash
python -m pytest tests/ -v
```

The tests use a small offline stand-in for the embedding model, so they need no internet
and no API key. They check that the code works, not how good the real model's results are.

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
