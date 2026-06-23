"""Example: run the documentation-QA agent on a question set.

Set the required environment variables first (or put them in a ``.env`` file):

    export DOCQA_AZURE_ENDPOINT="https://<resource>.openai.azure.com"
    export DOCQA_AZURE_API_KEY="<key>"
    export DOCQA_CHAT_DEPLOYMENT="gpt-oss-120b"
    export DOCQA_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
    export DOCQA_GITLAB_TOKEN="<token>"      # if cloning a private repo

Then:  poetry run python examples/run_example.py
"""

from __future__ import annotations

from doc_qa_agent import DocQAAgent, QuestionSet, get_settings


def main() -> None:
    """Build a question set, run the agent, and print calibrated answers."""
    settings = get_settings()

    question_set = QuestionSet(
        set_id="demo-1",
        repo_url="https://gitlab.com/group/project.git",
        requires_code_execution=True,  # the imposed binary flag, set per question set
        questions=[
            "What does the train_model function return?",
            "Run the example notebook and report the final accuracy.",
            "Which optimizer is used by default in the config?",
        ],
    )

    agent = DocQAAgent(settings)
    result = agent.run(question_set)

    print(f"set_id={result.set_id}  repo_indexed={result.repo_indexed}")
    for note in result.notes:
        print(f"  note: {note}")
    for qa in result.answers:
        print("\n" + "=" * 70)
        print(f"Q: {qa.question}")
        print(f"A: {qa.answer}")
        print(f"answerable={qa.answerable}  confidence={qa.confidence:.3f}")
        if qa.confidence_detail:
            s = qa.confidence_detail.signals
            print(
                f"   signals: verbalized={s.verbalized:.2f} "
                f"consistency={s.self_consistency} "
                f"evidence={s.evidence_strength:.2f} grounded={s.grounded}"
            )

    # Emit the strict output schema as JSON (what you'd submit).
    print("\n--- JSON output ---")
    print(result.model_dump_json(indent=2, exclude={"answers": {"__all__": {"evidence"}}}))


if __name__ == "__main__":
    main()
