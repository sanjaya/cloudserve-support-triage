# CloudServe support triage

Capstone project, Forward Deployed AI Engineering. Sanjaya Kumar Panigrahy.

CloudServe's help articles already answer about seven in ten support tickets, but more than
half of all tickets are still passed to senior staff. This system:

- answers a ticket from the help articles when it can, and shows which article each answer came from;
- passes every other ticket to a person, with a short summary and the useful articles attached;
- records every decision and the reason for it, so any answer can be explained later.

> Status: week two. Setup (B-01), ticket reading (B-02), the run harness (B-05), searching
> the help articles (B-03, B-04), classifying tickets (B-06), deciding (B-07), writing the
> answer (B-08) and the safety checks on answers (B-09) are done.

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

# choose the confidence threshold for automatic answers (development tickets; uses the
# evaluate_classifier.py results, so no model calls)
python scripts/sweep_threshold.py

# compare versions of the answer prompt PR-03 on development tickets (makes model calls; cached)
# (copy ground_truth_responses.json from Capstone_Pack/05_Datasets/ into data/ to also check must_not_claim)
python scripts/evaluate_answers.py

# test the safety checks with engineered cases (another customer's data, hidden instructions,
# an unsupported claim, a refund promise, a fix date) and two answers that must pass
python scripts/check_guardrails.py

# compare versions of the grounding check PR-05 on development answers (uses the
# evaluate_answers.py output; makes model calls, cached)
python scripts/evaluate_grounding.py

# retrain the classifier (only needed if the data or the embedding model changes)
python scripts/train_classifier.py

# full unattended run over any ticket file; results go to a dated folder inside --output
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/
```

The first run downloads the embedding model (about 90 MB) and builds the search store in
`storage/`. This happens automatically and takes a minute; later runs reuse both.

Right now the system classifies each ticket, searches the help articles, decides whether it
could be answered automatically (`src/route.py`) and, if so, writes the answer from the
passages with prompt PR-03 (`src/generate.py`). The answer is sent only if the model says the
articles fully answer the question and every passage it cites is one search found; otherwise a
person handles the ticket, with the draft attached.

Before an answer is sent, every safety check runs on it (`src/guardrails.py`, FR-07): the
never-automate list again; private data (email addresses, phone numbers, customer ids, keys,
other customers' names); promises of refunds, fixes or dates; instructions hidden in the ticket
(a code pattern plus prompt PR-06); and whether every claim is supported by the cited passage
(prompt PR-05). If any check blocks, or cannot run, the answer is not sent (action "block") and
a person handles the ticket. Every result is recorded in the run's results.

How the decision is made (FR-04, FR-05), first rule that applies wins: could not classify ->
a person; a never-automate topic (security incident, compliance request, feature request,
unclear request) -> a person, however sure; confidence below `CONFIDENCE_THRESHOLD` -> a person;
no help passage matched -> a person; otherwise -> answer automatically. The decision is plain
code, so the same ticket always gets the same decision.

How the steps are joined: the ticket flow is a LangGraph graph (`src/graph.py`). Classify and
search run side by side, then decide, then hand-off (or, from B-07, an answer). To see the graph:
`python -m src.graph` prints it as a Mermaid diagram. LangChain is used for the model call
(`ChatOpenAI` pointed at OpenRouter, prompts as `ChatPromptTemplate`) and for search
(`langchain-chroma` with `HuggingFaceEmbeddings`). See `docs/architecture.md`.

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
says where. Other people's libraries are listed in `requirements.txt`; the main ones are:

| Library | Used for | Where |
|---|---|---|
| LangGraph (LangChain Inc., MIT licence) | The ticket flow as a graph | `src/graph.py` |
| LangChain: langchain-core, langchain-openai, langchain-huggingface, langchain-chroma (MIT licence) | Prompt templates, the model call, embeddings, the vector store | `src/llm.py`, `src/retrieve.py` |
| Chroma (Apache 2.0) | Storing and searching the passages | through langchain-chroma |
| sentence-transformers, all-MiniLM-L6-v2 (Apache 2.0) | Turning text into embeddings | through langchain-huggingface |
| scikit-learn (BSD) | The trained classifier (logistic regression) | `src/classify.py` |
| OpenRouter, Meta Llama 3.1 8B Instruct | The language model | called from `src/llm.py` |
