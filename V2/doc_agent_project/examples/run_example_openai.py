"""Run the agent against a self-hosted gpt-oss-120b (OpenAI-compatible endpoint).

The ONLY differences from examples/run_example.py are in Settings: the provider,
the base_url, and the model id. Everything else — the agentic loop, tools, RAG,
and calibration — is identical, because the agent is provider-agnostic.

Serve gpt-oss-120b with an OpenAI-compatible server first, e.g. vLLM:
    vllm serve openai/gpt-oss-120b --enable-auto-tool-choice --tool-call-parser openai

Then:
    export DOC_AGENT_API_KEY=EMPTY          # many local servers accept any token
    poetry run python examples/run_example_openai.py
"""

from __future__ import annotations

from doc_agent import DocAgent, LocalExecutionBackend, QuestionItem, QuestionSet, Settings


def main() -> None:
    """Configure the OpenAI-compatible provider and answer a couple of questions."""
    settings = Settings(
        llm_provider="openai",
        model="openai/gpt-oss-120b",          # as your server names it
        base_url="http://localhost:8000/v1",  # your endpoint
        api_key="EMPTY",                       # or a real token if the provider needs one
        # Embeddings: gpt-oss is chat-only, so keep local SentenceTransformers for RAG.
        # To use a provider-hosted embeddings model instead, set:
        #   embedding_provider="openai",
        #   embedding_model="<provider-embedding-model-id>",
        #   embedding_base_url="http://localhost:8000/v1",
        embedding_provider="auto",
        enable_rag=True,
        max_agent_steps=8,
    )
    backend = LocalExecutionBackend(workdir=settings.workdir, timeout_seconds=120)
    agent = DocAgent(backend=backend, settings=settings)

    payload = QuestionSet(
        repo_url="https://github.com/psf/requests.git",
        questions=[
            QuestionItem(
                id="q1",
                question="What is the default timeout used by requests.get when none is passed?",
                requires_code_execution=False,
            ),
            QuestionItem(
                id="q2",
                question="What status_code is returned by get() against httpbin /status/204?",
                requires_code_execution=True,
            ),
        ],
    )

    result = agent.answer_questions(payload)
    for ans in result.answers:
        print("=" * 70)
        print(f"Q: {ans.question_id}")
        print(f"Answer: {ans.answer}")
        print(f"Confidence: {ans.confidence} ({ans.confidence_label.value})")
        print(f"Tools used: {ans.tools_used}")


if __name__ == "__main__":
    main()
