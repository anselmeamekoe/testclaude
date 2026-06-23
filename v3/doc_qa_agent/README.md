# doc-qa-agent

An **agentic** documentation-QA system for the hackathon challenge. It answers
questions about a repository by deciding — per question — whether to search
docs, search code, read files, or **execute code**, and it returns an answer
with a **calibrated confidence** (over- and under-confidence are both penalised).

It is not a chatbot: GPT-OSS-120 (served on Azure) drives a tool-calling loop
with an explicit decision policy and a dedicated calibration stage.

## Architecture

```
QuestionSet (pydantic, input)               # questions + requires_code_execution flag
        │
   DocQAAgent.run
        ├─ tools/repo.clone            → clone private GitLab repo (imposed signature)
        ├─ tools/execution.create_venv → venv (only if requires_code_execution)
        ├─ tools/retrieval.index_repository → FAISS code-index + doc-index (RAG)
        │
   for each question → tool-calling loop (GPT-OSS-120):
        search_docs · search_code · read_file
        execute_python_snippet/file · execute_notebook · install_packages   ← gated by the flag
        finish(answer, verbalized_confidence, answerable, evidence)
        │
   confidence.ConfidenceCalibrator                 # verbalized × self-consistency × evidence
        │
AgentResult (pydantic, output)              # per-question answer + calibrated confidence
```

## Why it should win

- **Knows *when* to do *what*.** The system prompt encodes the decision policy
  (docs for concepts, code for "where/how", execution only to observe
  behaviour). The imposed `requires_code_execution` flag physically **removes**
  the execution tools when false, so the model can't waste budget running code.
- **Knows when information is missing.** The model can set `answerable=false`;
  the calibrator then caps confidence low, dodging the over-confidence penalty.
- **Calibrated confidence, not vibes.** Final confidence fuses three signals in
  logit space: the model's verbalized probability, **self-consistency** across
  resampled answers, and **evidence strength** (retrieval scores + successful
  code execution + corroboration). Hard caps for ungrounded/unanswerable cases.
  `fit_temperature()` re-fits on a labelled dev set to minimise ECE before
  submission.
- **RAG built in.** FAISS (cosine) over separately-indexed code and docs, with
  an optional **Qdrant** ("sparrow") backend for persistence (imposed
  `get_qdrant_client`).
- **Strict Pydantic everywhere**, no dataclasses; every tool has a docstring.

## Confidence model (summary)

```
z = bias
  + w_verbalized · logit(verbalized)
  + w_consistency · (self_consistency − 0.5)·2
  + w_evidence    · (evidence_strength − 0.5)·2
p = sigmoid(z / temperature)
p = min(p, ungrounded_cap)   if answer not grounded in evidence
p = min(verbalized, unanswerable_cap)   if not answerable
```

Observed behaviour (offline test): strong+executed → ~0.96; confident but
**unsupported** → capped 0.45; unanswerable → 0.15.

## Install & run

```bash
poetry install
export DOCQA_AZURE_ENDPOINT="https://<resource>.openai.azure.com"
export DOCQA_AZURE_API_KEY="<key>"
export DOCQA_CHAT_DEPLOYMENT="gpt-oss-120b"
export DOCQA_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
export DOCQA_GITLAB_TOKEN="<token>"
poetry run python examples/run_example.py
```

If GPT-OSS-120 is behind an OpenAI-compatible gateway instead of native Azure:

```bash
export DOCQA_USE_OPENAI_COMPATIBLE_BASE_URL=true
export DOCQA_OPENAI_BASE_URL="https://<gateway>/v1"
```

For local smoke tests without Azure, construct the agent with
`DocQAAgent(settings, offline_embeddings=True)` to use the deterministic
hashing embedder.

## Layout

| File | Purpose |
|------|---------|
| `models.py` | Imposed input/output schemas + evidence/confidence models |
| `config.py` | `pydantic-settings` configuration |
| `llm.py` | Azure GPT-OSS-120 chat (tool calling) + embeddings |
| `agent.py` | The tool-calling reasoning loop |
| `confidence.py` | Calibration, temperature fitting, ECE |
| `prompts.py` | System prompt + tool schemas (flag-gated) |
| `tools/repo.py` | Imposed `clone()` |
| `tools/execution.py` | venv / install / run file / snippet / notebook |
| `tools/retrieval.py` | FAISS RAG (code & docs) + `RepoIndex` |
| `tools/storage.py` | Imposed Qdrant `get_qdrant_client` + vector store |

> Adjust the input/output Pydantic schemas to match the organizers' exact field
> names if they differ from `QuestionSet` / `AgentResult`.
