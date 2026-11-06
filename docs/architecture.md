# Design

Each ticket goes through six steps:

1. Read: put the ticket in one standard shape.
2. Classify: decide what the ticket is about and how urgent it is.
3. Search: find the help-article passages that match.
4. Decide: answer automatically, or send to a person.
5. Answer: write a reply using only those passages, and show the sources.
6. Check: block the reply if it is unsafe or not supported by the sources.

Every decision is recorded with its reason in the decision log (`src/logging_store.py`, B-10):
one record per step per ticket, written by the harness after each ticket, including tickets that
failed part-way. If any step fails, the ticket goes to a person.
More detail is added here as each step is built (see the Stage 3 specifications, FR-01 to FR-12).

## How the steps are joined: LangGraph

The Project Brief names "LangChain with LangGraph" for orchestration. The flow is a LangGraph
graph in `src/graph.py`; `pipeline.process()` runs one ticket through it, and the harness
calls `process()` for every ticket.

```mermaid
graph TD
    START([start]) --> check_input
    check_input -. has text .-> classify
    check_input -. has text .-> search
    check_input -. no text .-> hand_off
    classify --> decide
    search --> decide
    decide -. "a person (src/route.py rules)" .-> hand_off
    decide -. "automatic" .-> draft
    draft -. "usable answer" .-> check
    check -. "all checks pass" .-> reply
    check -. "any check blocks" .-> hand_off
    draft -. "not answered / bad citation" .-> hand_off
    reply --> END([end])
    hand_off --> END
```

- Classify and search do not depend on each other, so they run side by side; decide waits for both.
- Decide applies the routing rules in `src/route.py` (B-07): could not classify, a
  never-automate topic (FR-05), confidence below the threshold, or no matching passage ->
  a person; otherwise -> automatic.
- Draft writes the answer with prompt PR-03 (`src/generate.py`, B-08). Code then checks it: the
  model must say the articles fully answer the question, and every passage it cites must be one
  search found. Otherwise the ticket goes to a person with the draft attached. The line saying
  the reply was drafted automatically is added by code if missing.
- Check runs every safety check on the answer (`src/guardrails.py`, B-09): never-automate list
  again, private data, promises (code patterns), instructions in the ticket (code pattern and
  PR-06), and grounding (PR-05). All results are recorded; any block, or a check that could not
  run, stops the answer (action "block") and sends the ticket to a person.
- `python -m src.graph` prints the current graph as a Mermaid diagram.

## What LangChain does, and what is our own code

| Step | LangChain / LangGraph part | Our own code | Why our own code |
|---|---|---|---|
| Joining the steps | LangGraph `StateGraph` | the nodes in `src/graph.py` | – |
| Read & clean (B-02) | – | `src/ingest.py` | Plain data cleaning; nothing in LangChain does it |
| Splitting articles (B-03) | – | `split_by_section` in `src/retrieve.py` | Splitting by the article's own sections won the measured comparison |
| Search (B-04) | `langchain-chroma` (`Chroma`), `HuggingFaceEmbeddings` | minimum score, passage ids for citations | – |
| Model call (B-06 and later) | `ChatOpenAI` (OpenRouter), `ChatPromptTemplate`, messages | answer cache, waiting and retrying, counting calls | So runs are repeatable, stay inside rate limits, and the metrics can report the number of calls |
| Classify (B-06) | – | trained classifier, agreement cases, measured confidence | Our own measured design (Stage 5, change 2); LangChain has no part for it |
| Write the answer (B-08) | the model call (`ChatOpenAI`, `ChatPromptTemplate` from PR-03) | citation and status checks in `src/generate.py` | The checks must be plain code so a bad citation can never reach a customer |
| Safety checks (B-09) | the model calls for PR-05 and PR-06 | the private-data and promise patterns, and the rule that any block or error stops the answer | Code checks are exact and free; a check that cannot run must stop the answer, not let it through |
| Hand-off note (B-16) | the model call for PR-04 | intent, reason, confidence and articles filled by code; fallback note when the model is down | A person must be able to trust the facts in the note; a missing model must not mean a missing note |
| Failure handling (B-17) | – | retries with growing waits, circuit breaker, search-failure route, malformed-input handling | No failure may stop the run (FR-12, NFR-02) |
| Decide (B-07) | the conditional edge out of decide | the rules in `src/route.py` | Must be plain code so the same ticket always gets the same decision (FR-04) |

Prompts stay as versioned files in `prompts/`. `src/llm.py` turns each file into a LangChain
`ChatPromptTemplate`. The text sent is exactly the same as before the move, so saved answers
in the cache still match.

## Search settings

Chroma's search is approximate (HNSW). Its default setting checks too few candidates when we
ask for 10 passages, and could miss close ones. `INDEX_SETTINGS` in `src/retrieve.py` raises it
so the result matches checking every passage (tested in `tests/test_retrieve.py`).
