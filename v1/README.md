# doc-agent — agentic documentation QA with calibrated confidence

An **agentic reasoning system** (not a chatbot) that answers documentation
questions about a repository by deciding, per question, when to **search code**,
**search docs**, **execute code**, or declare **information missing** — and then
reports a **calibrated confidence** score where both over- and under-confidence
are penalized.

It is built around the execution tools the organizers provide (clone GitLab repo,
create venv, install packages, run files/notebooks, execute snippets) and adds the
reasoning, retrieval, and calibration layers on top.

## How it maps to the five required decisions

| Decision | Where it lives |
|---|---|
| 1. When to search code | `search_code` tool (`tools/retrieval.py`) + policy in `prompts.py` |
| 2. When to search docs | `search_docs` tool (`tools/retrieval.py`) + policy in `prompts.py` |
| 3. When to execute code | execution toolchain (`tools/execution.py`), prioritized when a question's `requires_code_execution` flag is `True` |
| 4. When information is missing | `submit_answer(information_complete=False, missing_information=...)`, enforced in the calibrator |
| 5. How confident it should be | `confidence.py` — blends model self-report, retrieval support, and execution outcome into a calibrated score |

## Architecture

```
QuestionSet (pydantic)               # repo_url + list[QuestionItem]
        │                            # each item has the binary requires_code_execution flag
        ▼
DocAgent.answer_questions()          # agent.py — the orchestrator
        │
        ├─ clone repo (once)  ──────► ExecutionBackend (organizer sandbox / local)
        ├─ build FAISS index (once) ─► rag/ (chunking → embeddings → VectorIndex)
        │
        └─ per question: bounded tool-use loop
                 ├─ search_code / search_docs        (RAG)
                 ├─ setup_environment / run_python_file / run_notebook / execute_code
                 ├─ submit_answer  (terminal)
                 ▼
           ConfidenceSignals ──► ConfidenceCalibrator ──► AgentAnswer (calibrated)
```

The data contract lives in `models.py`. Everything in and out of the agent is a
validated pydantic model.

### The binary `requires_code_execution` flag

Each `QuestionItem` carries this flag from the organizers. The agent uses it as a
strong routing prior (`prompts.build_question_prompt`):

- **`True`** → the agent is told to set up the environment and *run* the relevant
  code to verify the answer. The calibrator then **caps confidence at 0.50** unless
  execution actually ran and succeeded — so an execution-required answer derived only
  from reading the source stays appropriately modest.
- **`False`** → the agent leads with retrieval and only executes if reading the
  source and docs cannot settle the answer.

## Confidence calibration

The model's self-reported confidence is **one input, not the answer** — LLMs are
systematically over-confident. `ConfidenceCalibrator` blends three signals:

1. **Verbalized confidence** from `submit_answer`.
2. **Retrieval support** — quality + coverage of the evidence actually retrieved
   (0 when nothing was found, which correctly drags confidence down).
3. **Execution signal** — verified-by-clean-run ranks far above inferred-from-reading;
   required-but-not-run is penalized.

Hard rules layered on the blend:
- `information_complete=False` ⇒ confidence floored to ~0.15.
- execution required & verified ⇒ may reach high confidence (cap 0.97).
- execution required & unverified ⇒ cap 0.50.
- read-only ⇒ cap 0.92 (never claim execution-level certainty without execution).

An optional `PlattScaler` can be **fit on labeled dev data** (`scaler.fit(probs,
labels)`) to remove residual bias; until fit it is the identity, so the system is
calibrated-by-construction and improves if you provide ground truth.

## RAG (optional, FAISS)

`rag/` chunks the repo (Python by symbol via `ast`; docs by section), embeds with
SentenceTransformers, and indexes with FAISS. If `faiss` / `sentence-transformers`
are not installed it transparently falls back to a numpy brute-force index and a
deterministic hashing embedder, so the pipeline always runs. Toggle with
`Settings.enable_rag`.

## Install

```bash
poetry install                 # core (numpy fallback RAG)
poetry install --extras rag    # FAISS + sentence-transformers (recommended)
poetry install --extras all    # + jupyter for run_notebook
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
poetry run python examples/run_example.py   # real run (needs API key + network)
poetry run python examples/smoke_test.py    # offline pipeline test (no API key)
poetry run pytest                           # unit tests
```

## Plugging in the organizer execution tools

`LocalExecutionBackend` is a working reference using `git`/`venv`/`subprocess`.
For the competition, implement the same `ExecutionBackend` protocol as a thin
adapter over the organizer API — nothing else changes:

```python
from doc_agent.tools import ExecutionBackend
from doc_agent.models import ExecutionResult

class OrganizerBackend(ExecutionBackend):
    def clone_repo(self, repo_url): ...
    def create_venv(self, repo_path): ...
    def install_packages(self, repo_path, packages): ...
    def run_python_file(self, repo_path, file_path, args): ...
    def run_notebook(self, repo_path, notebook_path): ...
    def execute_snippet(self, repo_path, code): ...

agent = DocAgent(backend=OrganizerBackend(), settings=Settings())
```

## Layout

```
doc_agent/
  models.py        # pydantic contract (QuestionItem.requires_code_execution lives here)
  config.py        # Settings
  llm.py           # LLMClient protocol + Anthropic tool-use client
  prompts.py       # system prompt, per-question routing policy, submit_answer schema
  agent.py         # DocAgent orchestrator (the agentic loop)
  confidence.py    # ConfidenceCalibrator + PlattScaler
  rag/             # chunking, embeddings, FAISS index
  tools/           # base framework, retrieval tools, execution tools
examples/          # run_example.py (real), smoke_test.py (offline)
tests/             # pytest unit tests
```
