# CloudServe support triage

[![CI](https://github.com/sanjaya/cloudserve-support-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/sanjaya/cloudserve-support-triage/actions/workflows/ci.yml)

Capstone project, Forward Deployed AI Engineering. Sanjaya Kumar Panigrahy.

CloudServe's help articles already answer about seven in ten support tickets, but more than
half of all tickets are still passed to senior staff. This system:

- answers a ticket from the help articles when it can, and shows which article each answer came from;
- passes every other ticket to a person, with a short summary and the useful articles attached;
- records every decision and the reason for it, so any answer can be explained later.

> Status: build complete. Done: B-01 to B-14, B-16, B-17 and B-19 to B-21. The gate run
> (B-11) passed all 9 checks on the 80 validation tickets and on a 120-ticket rehearsal of the
> hidden set. Governance: kill switch (B-21) and fairness audit (B-14). The human review kit
> (B-22) is ready; the two-person human review is requested from the capstone reviewer.
> Next: report and video (B-15).
> Also done (Should items): monitoring (B-12) and a single-ticket API (B-23). Not done:
> urgency ordering of the queue (B-18).

## Assessor quick start (clean checkout, about 15 minutes)

These are the exact steps to repeat the gate run on a fresh machine. Nothing else is needed:
the trained classifier and its measured confidence (`models/`) and the help articles and
validation tickets (`data/`) are in the repository.

```bash
git clone https://github.com/sanjaya/cloudserve-support-triage.git
cd cloudserve-support-triage
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip && python -m pip install -r requirements.txt
cp .env.example .env                       # put your OpenRouter key in OPENROUTER_API_KEY
python scripts/check_model.py              # 'ready'
python -m pytest tests/ -q                 # all pass, no key or internet needed

# the hidden set: give its path with --input (JSON array or JSON Lines both work)
python -m evaluation.harness --input <hidden_tickets_file> --output evaluation/results/
python scripts/check_gate.py --run evaluation/results/run_<...> --input <hidden_tickets_file>
```

The harness writes `results.jsonl`, `metrics.json` and `metrics.md` into a new dated folder,
and one record per step per ticket into the decision log. `check_gate.py` writes
`gate_check.md` next to them and exits with 1 if any check fails. The model calls are paced
and cached, so about 120 tickets take around 35 minutes the first time and seconds when
repeated. If the provider fails part-way, the run still finishes: affected tickets go to a
person with the reason.

To try it without a key: set `LLM_DISABLED=1`. Every ticket then goes to a person, with a
hand-off note written by code (this is what CI runs on every push).

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

# live monitoring: start Prometheus and Grafana, then run with the metrics switched on
# (details and the no-Docker route in monitoring/README.md)
docker compose -f monitoring/docker-compose.yml up -d
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/ --metrics-port 8001 --pace 2 --keep-serving

# gate check (B-11): verify a finished run against acceptance criteria A9, A10 and A8
python scripts/check_gate.py --run evaluation/results/run_<...> --input data/validation_tickets.json

# rehearse the hidden set: 120 unlabelled tickets as JSON Lines, at a path outside the repo
python scripts/make_rehearsal_set.py --out ~/workspace/rehearsal/rehearsal_120.jsonl
python -m evaluation.harness --input ~/workspace/rehearsal/rehearsal_120.jsonl --output evaluation/results/

# failure test: cut the model provider off part-way through a real run, plus broken tickets;
# checks the run still finishes with every ticket logged (FR-12, A11). Costs nothing.
python scripts/induce_failures.py

# see how one ticket was handled, step by step, from the decision log (latest run)
python -m src.logging_store --ticket VAL-0001

# KILL SWITCH (B-21): stop all automatic answers at once, no restart needed (takes effect on
# the next ticket, even in a run that is already going). Every change is logged with who and why.
python -m src.kill_switch on --by "Marcus Adeyemi" --reason "wrong answers reported"
python -m src.kill_switch status
python -m src.kill_switch off --by "Marcus Adeyemi"

# fairness audit (B-14, NFR-06): compare resolution and quality across tier, region, fluency,
# ticket length and channel for a finished run (no model calls)
python scripts/fairness_audit.py --run evaluation/results/run_<...> --input data/validation_tickets.json \
    --dev-classifier evaluation/results/classifier_per_ticket.csv

# human review of 50 answers (B-22, NFR-03): make the two-reviewer workbook (about 50 model calls
# for the automatic judge PR-08), have two people fill it in, then score it
python scripts/prepare_review.py --runs evaluation/results/run_<gate> evaluation/results/run_<rehearsal>
python scripts/score_review.py --review evaluation/results/review_<...>

# single-ticket API (B-23): the same steps for one ticket at a time, e.g. for a live demo.
# Then open http://127.0.0.1:8000/docs and use "Try it out" on POST /tickets.
python -m src.api
curl -X POST http://127.0.0.1:8000/tickets -H "Content-Type: application/json" \
     -d '{"ticket_id": "DEMO-1", "channel": "email", "subject": "Webhooks failing", "body": "Signature verification fails for every delivery. We compute it over the parsed body."}'
curl http://127.0.0.1:8000/tickets/DEMO-1        # how it was handled, from the decision log
curl http://127.0.0.1:8000/health                # up? kill switch? model and threshold

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

Every ticket that goes to a person gets a hand-off note (`src/escalate.py`, FR-08): a short
summary, intent, urgency and confidence, what the customer has already told us, the articles
search found, what the system was unsure about, the reason, a suggested first question and any
draft. Prompt PR-04 writes the summary parts; code fills the rest, so the model cannot change the
intent, the reason or the articles. If the model is down, code writes a plain note instead.

When something breaks (FR-12), the ticket goes to a person with the reason, and the run carries
on: a timed-out or rate-limited model call is retried with growing waits; after repeated failures
a circuit breaker stops calling the provider for a minute so the remaining tickets are not each
held up (answers already in the cache still work); a failed search, or a broken ticket record,
is handled the same way. `scripts/induce_failures.py` proves it in a real run.

Every decision is also written to the decision log (`src/logging_store.py`, FR-09): a database
table at `DATABASE_URL` (SQLite `storage/decisions.db` by default, not committed) with one record
per step per ticket – classify, search, route, draft, safety, final – each with an input summary,
model, prompt version, prediction, confidence, sources, threshold, action, plain-language reason,
safety-check results and requirement ids. The metrics report checks that the number of tickets
logged equals the number processed ("decision log reconciles").

How the decision is made (FR-04, FR-05), first rule that applies wins: could not classify ->
a person; a never-automate topic (security incident, compliance request, feature request,
unclear request) -> a person, however sure; confidence below `CONFIDENCE_THRESHOLD` -> a person;
no help passage matched -> a person; otherwise -> answer automatically. The decision is plain
code, so the same ticket always gets the same decision.

How the steps are joined: the ticket flow is a LangGraph graph (`src/graph.py`). Classify and
search run side by side, then decide, then either the answer (draft, safety checks, reply) or the hand-off note. To see the graph:
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

# check that no key or token is in the repository or its history (CI runs this on every push)
python scripts/check_no_secrets.py --history
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
| `.github/workflows/ci.yml` | Continuous integration on every push: key scan, tests, and a 6-ticket pipeline run with the gate check (B-13) |
| `monitoring/` | Prometheus and Grafana set-up and the dashboard (B-12) |
| `models/` | The trained classifier and its measured confidence (committed) |
| `data/` | Help articles and validation tickets only (see `data/README.md`) |
| `scripts/` | Small helper scripts, including the Stage 1 data analysis |

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
