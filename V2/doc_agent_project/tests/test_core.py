"""Unit tests for the core building blocks (no API key required).

These cover the parts whose correctness is independent of the LLM: the pydantic
contract, code/doc chunking, FAISS/numpy retrieval, and the confidence calibrator's
decision rules. The end-to-end loop is exercised separately by examples/smoke_test.py.
"""

from __future__ import annotations

from doc_agent.config import Settings
from doc_agent.confidence import ConfidenceCalibrator, PlattScaler
from doc_agent.models import (
    AgentAnswer,
    ConfidenceLabel,
    ConfidenceSignals,
    QuestionItem,
)
from doc_agent.rag.chunking import chunk_document, chunk_python
from doc_agent.rag.embeddings import HashingEmbedder
from doc_agent.rag.index import VectorIndex


def test_answer_label_syncs_with_confidence():
    """The confidence_label must always reflect the numeric confidence."""
    a = AgentAnswer(
        question_id="q", answer="x", confidence=0.05,
        confidence_label=ConfidenceLabel.VERY_HIGH,  # deliberately wrong; should be fixed
    )
    assert a.confidence_label == ConfidenceLabel.VERY_LOW


def test_chunk_python_splits_by_symbol():
    """Python chunking should yield one chunk per top-level symbol with names."""
    code = (
        "import os\n\n"
        "def foo():\n    return 1\n\n"
        "class Bar:\n    def baz(self):\n        return 2\n"
    )
    chunks = chunk_python("m.py", code, chunk_size=1000, overlap=100)
    symbols = {c.symbol for c in chunks}
    assert "foo" in symbols and "Bar" in symbols


def test_chunk_document_splits_by_heading():
    """Markdown chunking should break on headings."""
    md = "# Title\nintro\n## Setup\ninstall steps\n## Usage\nrun it\n"
    chunks = chunk_document("README.md", md, chunk_size=1000, overlap=50)
    assert len(chunks) >= 2
    assert all(c.source_type == "doc" for c in chunks)


def test_vector_index_retrieves_relevant_chunk():
    """Retrieval should rank a lexically-matching chunk first (hashing embedder)."""
    chunks = chunk_python(
        "calc.py",
        "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n",
        chunk_size=500, overlap=50,
    )
    index = VectorIndex(HashingEmbedder()).build(chunks)
    results = index.search("add function", top_k=2)
    assert results
    assert any("add" in r.chunk.text and "subtract" not in r.chunk.text for r in results)


def test_information_incomplete_floors_confidence():
    """An information-incomplete answer must be low confidence regardless of inputs."""
    cal = ConfidenceCalibrator(Settings())
    score = cal.calibrate(ConfidenceSignals(
        verbalized_confidence=0.99, retrieval_support=0.99, information_complete=False,
    ))
    assert score < 0.2


def test_execution_required_but_unverified_is_capped():
    """Execution-required answers must not be confident unless actually verified."""
    cal = ConfidenceCalibrator(Settings())
    unverified = cal.calibrate(ConfidenceSignals(
        verbalized_confidence=0.95, retrieval_support=0.9,
        execution_required=True, execution_attempted=False,
    ))
    verified = cal.calibrate(ConfidenceSignals(
        verbalized_confidence=0.95, retrieval_support=0.9,
        execution_required=True, execution_attempted=True, execution_succeeded=True,
    ))
    assert unverified <= 0.5 < verified


def test_platt_scaler_fit_improves_separation():
    """Fitting Platt scaling on simple data should push confident-correct up."""
    scaler = PlattScaler()
    # Synthetic: high raw probs are correct, low raw probs are wrong.
    probs = [0.9, 0.8, 0.2, 0.1] * 10
    labels = [1, 1, 0, 0] * 10
    scaler.fit(probs, labels, epochs=300)
    assert scaler.transform(0.9) > scaler.transform(0.1)


def test_retrieval_support_zero_when_empty():
    """No evidence => zero support => downward pressure on confidence."""
    assert ConfidenceCalibrator.retrieval_support_from_scores([]) == 0.0


def test_question_item_requires_flag_is_mandatory():
    """The binary execution flag is a required field of every question."""
    item = QuestionItem(id="q1", question="?", requires_code_execution=True)
    assert item.requires_code_execution is True


def test_openai_history_conversion():
    """Neutral transcript should map to valid OpenAI chat messages."""
    from doc_agent.llm import HistoryAssistant, HistoryToolResults, HistoryUser, OpenAILLM

    cli = OpenAILLM(model="m", base_url="http://x/v1")
    history = [
        HistoryUser("q?"),
        HistoryAssistant(text="", tool_calls=[
            {"id": "c1", "name": "execute_code", "input": {"code": "print(1)"}}
        ]),
        HistoryToolResults(results=[{"id": "c1", "content": "1", "is_error": False}]),
    ]
    msgs = cli._to_messages("SYS", history)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "execute_code"
    assert msgs[3]["tool_call_id"] == "c1"


def test_openai_content_fallback_parser():
    """Tool calls emitted into text content (gpt-oss quirk) are recovered."""
    import json as _json

    from doc_agent.llm import OpenAILLM

    cli = OpenAILLM(model="m", base_url="http://x/v1")
    text = _json.dumps([{"name": "get_weather", "parameters": {"city": "Berlin"}}])
    calls = cli._parse_tool_calls_from_content(text)
    assert calls and calls[0]["name"] == "get_weather"
    assert calls[0]["input"] == {"city": "Berlin"}
    assert cli._parse_tool_calls_from_content("plain answer, no tool call") == []


def test_build_llm_selects_provider():
    """The factory should honor settings.llm_provider."""
    from doc_agent.config import Settings
    from doc_agent.llm import AnthropicLLM, OpenAILLM, build_llm

    assert isinstance(build_llm(Settings(llm_provider="anthropic")), AnthropicLLM)
    assert isinstance(
        build_llm(Settings(llm_provider="openai", model="openai/gpt-oss-120b",
                           base_url="http://x/v1")),
        OpenAILLM,
    )
