# Sparrow Documentation-QA Agent

An **agentic reasoning system** (not a chatbot) that answers documentation
questions about a code repository by deciding, per question, when to **search
code**, **search docs**, **execute code**, or **abstain** — and that reports a
**calibrated** confidence (over- and under-confidence are both penalised).

Uses **gpt-oss-120b** for reasoning + tool calling and an **OpenAI embedding
model** for RAG, both via OpenAI-compatible endpoints. All data models are
**Pydantic** (no dataclasses).

## Why this can win

The agent is built around the five competencies the brief calls out:

| Competency             | How it's implemented                                                                 |
|------------------------|--------------------------------------------------------------------------------------|
| When to search code    | `search_code` tool over a FAISS index of AST-split Python + other source            |
| When to search docs    | `search_docs` tool over README/.md/.rst/docstring chunks                            |
| When to execute code   | Execution tools exposed **only** when `Input.code_execution == True`; agent decides |
| When info is missing    | `submit_answer(not_known=true)` — the prompt rewards abstaining over guessing        |
| How confident to be    | A dedicated **calibration engine** fusing 4 signals (below)                          |

## Architecture

```
Input ──► clone repo ──► (venv if code_execution) ──► chunk+embed+index (FAISS)
                                                              │
        ┌─────────────────────── per question ───────────────┘
        ▼
   DocQAAgent loop (gpt-oss-120b + tools)
     search_code / search_docs / read_file / list_files
     execute_python_snippet|file|notebook / install_packages   (gated)
     submit_answer  ◄── terminal tool, maps 1:1 to AnswerItem
        ▼
   CalibrationEngine ──► calibrated confidence bucket + evidence ──► Output
```

| Module                       | Responsibility                                                  |
|------------------------------|-----------------------------------------------------------------|
| `config.py`                  | Pydantic-settings (two endpoints: chat vs. embeddings)          |
| `models.py`                  | All Pydantic models (contract + internal)                       |
| `llm.py`                     | gpt-oss-120b chat client (tool calling, `reasoning_effort`)     |
| `embeddings.py`              | OpenAI embeddings (batched, L2-normalised)                      |
| `rag.py`                     | Code-aware chunking + FAISS index + retriever                   |
| `tools.py`                   | Tool schemas, dispatch, and the execution-backend adapter       |
| `calibration.py`             | Confidence calibration engine                                   |
| `agent.py`                   | The per-question tool-calling loop (+ optional self-consistency)|
| `pipeline.py`                | clone → venv → index → answer all → `Output`, plus the CLI      |

## Confidence calibration

`submit_answer` carries the model's *self-reported* confidence, which is **not
trusted directly**. `CalibrationEngine.calibrate` fuses signals into a score in
`[0,1]`, then buckets it (`>=0.72` high, `>=0.45` medium, else low):

1. **Self-report** prior (low/med/high → 0.30/0.60/0.85).
2. **Evidence strength** — best retrieval cosine similarity; successful **code
   execution** counts as strong evidence (0.85 floor).
3. **Verifier** — an independent gpt-oss pass grades how well the gathered
   evidence actually supports the answer.
4. **Self-consistency** (optional) — answer N times, cluster answers by
   embedding similarity, use agreement as a signal.

Abstentions (`not_known=true`) are forced to **low**. The `evidence` field is
filled from what *actually happened* during the run, not just what the model
claimed.

## Setup

```bash
poetry install
cp .env.example .env   # fill in endpoints + keys
```

`.env` keys (all prefixed `SPARROW_`): `LLM_BASE_URL`, `LLM_API_KEY`,
`LLM_MODEL=gpt-oss-120b`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`,
`EMBEDDING_MODEL=text-embedding-3-small`, plus behaviour flags
(`ENABLE_VERIFIER`, `ENABLE_SELF_CONSISTENCY`, `MAX_AGENT_STEPS`, ...).

## Wiring your provided tools

The agent **reuses your existing** `tools/` helpers. Point two factory functions
at your real modules (defaults already match the layout you described):

* `sparrow_agent/tools.py → build_default_backend()`
  imports `execute_python_snippet/file/notebook` from `tools.code.code_execution`
  and `install_packages` from `tools.code.env_setup`.
* `sparrow_agent/pipeline.py → _build_provisioner()`
  imports `clone` from `tools.gitlab.clone` and `create_venv` from
  `tools.code.env_setup`.

If your import paths differ, edit those two functions (each is a few lines).

## Run

```bash
# from a file
poetry run sparrow-agent --input payload.json --output result.json

# or from stdin
cat payload.json | poetry run sparrow-agent
```

Programmatically:

```python
from sparrow_agent import AgentPipeline, Input

payload = Input.model_validate_json(open("payload.json").read())
output = AgentPipeline.from_settings().run(payload)
print(output.model_dump_json(indent=2))
```

## Notes

* Execution env vars (`SPARROW_TOKEN`, `DATASET_ACCESS_KEY`, ...) are injected
  into executed code so notebooks/scripts can authenticate to Sparrow/the dataset.
* `tiktoken` is loaded lazily with a character-count fallback, so chunking still
  works in locked-down environments where its BPE file can't be downloaded.
* The local FAISS index is in-process per submission; swap in your Qdrant client
  (`tools/vector_db/client.py`) inside `rag.VectorIndex` if you prefer a server.
