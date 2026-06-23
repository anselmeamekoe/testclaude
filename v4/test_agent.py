"""Offline tests for the agent. No network: LLM and embeddings are mocked.

Run with:  poetry run pytest -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sparrow_agent.agent import DocQAAgent
from sparrow_agent.calibration import CalibrationEngine, _weighted_mean
from sparrow_agent.config import Settings
from sparrow_agent.models import AnswerItem, TemplateQuestion, Trajectory
from sparrow_agent.rag import CodeChunker, Retriever
from sparrow_agent.tools import ToolBox, _CallableBackend


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    """Deterministic hash-based embedder so tests are reproducible offline."""

    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.standard_normal(self.dim).astype("float32")
        if "timeout" in text.lower():
            vec[0] += 5.0
        return vec

    def embed(self, texts, normalize=True):
        if not texts:
            return np.zeros((0, self.dim), "float32")
        matrix = np.stack([self._vec(t) for t in texts])
        if normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            matrix = matrix / norms
        return matrix

    def embed_one(self, text, normalize=True):
        return self.embed([text], normalize)[0]


class _Fn:
    def __init__(self, name, args):
        self.name = name
        self.arguments = json.dumps(args)


class _Call:
    def __init__(self, cid, name, args):
        self.id = cid
        self.function = _Fn(name, args)


class _Msg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class ScriptedLLM:
    """Replays a list of pre-defined tool-call turns, then verifier JSON."""

    def __init__(self, turns, support=0.9):
        self._turns = turns
        self._i = 0
        self._support = support

    def chat(self, messages, tools=None, tool_choice=None, temperature=None):
        turn = self._turns[min(self._i, len(self._turns) - 1)]
        self._i += 1
        return _Msg(tool_calls=turn)

    def complete_json(self, system, user, temperature=0.0):
        return {"support": self._support, "should_abstain": self._support < 0.4}


@pytest.fixture()
def repo(tmp_path) -> Path:
    (tmp_path / "config.py").write_text(
        "DEFAULT_TIMEOUT = 30\n\ndef get_timeout():\n"
        '    """Return the default request timeout."""\n    return DEFAULT_TIMEOUT\n'
    )
    (tmp_path / "README.md").write_text("# App\nThe default timeout is 30 seconds.\n")
    return tmp_path


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_python_chunker_splits_by_definition(repo):
    chunks = CodeChunker(Settings()).chunk_repo(repo)
    paths = {c.path for c in chunks}
    assert "config.py" in paths and "README.md" in paths
    # The function definition should be its own code chunk.
    assert any(c.kind == "code" and "get_timeout" in c.text for c in chunks)


def test_weighted_mean_basic():
    assert _weighted_mean([]) == 0.5
    assert _weighted_mean([(1.0, 1.0), (0.0, 1.0)]) == 0.5
    assert _weighted_mean([(1.0, 3.0), (0.0, 1.0)]) == 0.75


def test_calibration_forces_low_on_abstain():
    engine = CalibrationEngine(Settings(enable_verifier=False), llm=None)  # verifier off
    item = AnswerItem(question=1, answer="unknown", confidence="high", not_known=True)
    result = engine.calibrate("q", item, Trajectory(question_id=1), hits=[])
    assert result.bucket == "low"


def test_execution_evidence_raises_confidence():
    engine = CalibrationEngine(Settings(enable_verifier=False), llm=None)
    item = AnswerItem(question=1, answer="30", confidence="medium", not_known=False)
    traj = Trajectory(question_id=1, used_execution=True)
    result = engine.calibrate("q", item, traj, hits=[])
    assert result.bucket in {"medium", "high"}
    assert result.score >= 0.6


def test_agent_search_then_submit(repo):
    settings = Settings(enable_verifier=True, enable_self_consistency=False)
    embedder = FakeEmbedder()
    llm = ScriptedLLM(
        turns=[
            [_Call("c1", "search_code", {"query": "default timeout"})],
            [
                _Call(
                    "c2",
                    "submit_answer",
                    {
                        "answer": "30 seconds",
                        "confidence": "high",
                        "evidence": ["files"],
                        "not_known": False,
                    },
                )
            ],
        ],
        support=0.9,
    )
    retriever = Retriever(settings, embedder)
    retriever.build(str(repo))
    toolbox = ToolBox(retriever=retriever, repo_root=str(repo), code_execution=False)
    agent = DocQAAgent(settings, llm, embedder, CalibrationEngine(settings, llm))
    answer = agent.answer(TemplateQuestion(id=9, question="timeout?"), toolbox, "App")
    assert answer.question == 9
    assert answer.not_known is False
    assert answer.evidence == ["files"]
    assert answer.confidence in {"low", "medium", "high"}


def test_execution_tools_hidden_when_disabled(repo):
    retriever = Retriever(Settings(), FakeEmbedder())
    retriever.build(str(repo))
    tb_off = ToolBox(retriever=retriever, repo_root=str(repo), code_execution=False)
    names_off = {t["function"]["name"] for t in tb_off.schemas()}
    assert "execute_python_snippet" not in names_off

    backend = _CallableBackend(
        execute_python_snippet=lambda **k: "ok",
        execute_python_file=lambda **k: "ok",
        execute_notebook=lambda **k: "ok",
        install_packages=lambda **k: "ok",
    )
    tb_on = ToolBox(
        retriever=retriever, repo_root=str(repo), code_execution=True, backend=backend
    )
    names_on = {t["function"]["name"] for t in tb_on.schemas()}
    assert "execute_python_snippet" in names_on
