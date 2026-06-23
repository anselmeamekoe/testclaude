"""The agentic reasoning loop.

:class:`DocQAAgent` ties everything together:

1. Acquire the repository (clone), optionally create a venv, and build the
   FAISS code/doc indices.
2. For each question, run a tool-calling loop where GPT-OSS-120 decides whether
   to search docs, search code, read files, or execute code — gated by the
   ``requires_code_execution`` flag on the input.
3. Convert the model's self-assessment plus collected evidence and a
   self-consistency check into a **calibrated** confidence.

The agent is deliberately conservative: when evidence is thin it reports low
confidence (and ``answerable=false``) rather than guessing, which is what the
over/under-confidence scoring rewards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .confidence import ConfidenceCalibrator, evidence_strength
from .config import Settings
from .llm import LLMClient
from .models import (
    AgentResult,
    ConfidenceSignals,
    Evidence,
    EvidenceKind,
    FinalAnswerPayload,
    QuestionAnswer,
    QuestionSet,
    ToolCallRecord,
)
from .prompts import SYSTEM_PROMPT, build_tool_schemas
from .tools import execution
from .tools.repo import clone
from .tools.retrieval import (
    AzureEmbedder,
    HashingEmbedder,
    RepoIndex,
    index_repository,
)

_MAX_TOOL_PREVIEW = 4000  # chars of tool output fed back to the model


class _ToolDispatcher:
    """Executes the tool calls the model requests and records evidence.

    Args:
        index: The repository index (provides search + venv path). May be
            ``None`` for repo-less, purely conceptual questions.
        settings: For top-k and other retrieval limits.
    """

    def __init__(self, index: RepoIndex | None, settings: Settings) -> None:
        self._index = index
        self._s = settings
        self.evidence: list[Evidence] = []
        self.records: list[ToolCallRecord] = []

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        """Run one tool and return a string result for the model to read.

        Args:
            name: Tool name requested by the model.
            args: Parsed tool arguments.

        Returns:
            A textual result (truncated) appended to the conversation.
        """
        try:
            result, evidence = self._run(name, args)
        except Exception as exc:  # noqa: BLE001 — surface tool errors to the model
            result, evidence = f"[tool error] {exc}", []
            self.records.append(ToolCallRecord(name=name, arguments=args,
                                               result_preview=result[:300], ok=False))
            return result

        self.evidence.extend(evidence)
        self.records.append(
            ToolCallRecord(name=name, arguments=args, result_preview=result[:300], ok=True)
        )
        return result[:_MAX_TOOL_PREVIEW]

    def _run(self, name: str, args: dict[str, Any]) -> tuple[str, list[Evidence]]:
        """Internal switch mapping tool names to implementations."""
        if name == "search_docs":
            return self._search("docs", args["query"])
        if name == "search_code":
            return self._search("code", args["query"])
        if name == "read_file":
            return self._read_file(args)
        if name == "execute_python_snippet":
            return self._exec_snippet(args["code"])
        if name == "execute_python_file":
            return self._exec_file(args["file_path"])
        if name == "execute_notebook":
            return self._exec_notebook(args["notebook_path"])
        if name == "install_packages":
            venv = self._index.venv_python_path if self._index else None
            return execution.install_packages(args["packages"], venv or ""), []
        return f"[unknown tool: {name}]", []

    def _search(self, which: str, query: str) -> tuple[str, list[Evidence]]:
        """Run a code or docs semantic search and format hits for the model."""
        if self._index is None:
            return "[no repository indexed]", []
        hits = (
            self._index.search_docs(query, self._s.top_k)
            if which == "docs"
            else self._index.search_code(query, self._s.top_k)
        )
        if not hits:
            return f"[no {which} results]", []
        formatted = "\n\n".join(
            f"[{h.source} | score={h.score:.2f}]\n{h.content[:800]}" for h in hits
        )
        return formatted, hits

    def _read_file(self, args: dict[str, Any]) -> tuple[str, list[Evidence]]:
        """Read a (range of a) file, returning its content as evidence."""
        if self._index is None:
            return "[no repository]", []
        path = Path(args["path"])
        if not path.is_absolute():
            path = Path(self._index.repo_path) / path
        if not path.is_file():
            return f"[file not found: {path}]", []
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        start = max(0, int(args.get("start_line", 1)) - 1)
        end = int(args.get("end_line", len(lines)))
        snippet = "\n".join(lines[start:end])
        ev = Evidence(kind=EvidenceKind.FILE, source=f"{path}:{start + 1}-{end}",
                      content=snippet, score=0.6)
        return snippet[:_MAX_TOOL_PREVIEW], [ev]

    def _exec_snippet(self, code: str) -> tuple[str, list[Evidence]]:
        """Execute a snippet in the repo venv and record success/failure."""
        venv = self._index.venv_python_path if self._index else None
        out = execution.execute_python_snippet(code, venv)
        ok = out.rstrip().endswith("0")
        ev = Evidence(kind=EvidenceKind.EXECUTION, source="executed snippet",
                      content=out, success=ok, score=1.0 if ok else 0.0)
        return out, [ev]

    def _exec_file(self, file_path: str) -> tuple[str, list[Evidence]]:
        """Execute a repository .py file and record success/failure."""
        venv = self._index.venv_python_path if self._index else None
        if self._index and not Path(file_path).is_absolute():
            file_path = str(Path(self._index.repo_path) / file_path)
        out = execution.execute_python_file(file_path, venv)
        ok = out.rstrip().endswith("0")
        ev = Evidence(kind=EvidenceKind.EXECUTION, source=f"executed {file_path}",
                      content=out, success=ok, score=1.0 if ok else 0.0)
        return out, [ev]

    def _exec_notebook(self, nb_path: str) -> tuple[str, list[Evidence]]:
        """Execute a notebook and record the run as execution evidence."""
        venv = self._index.venv_python_path if self._index else None
        if self._index and not Path(nb_path).is_absolute():
            nb_path = str(Path(self._index.repo_path) / nb_path)
        out = execution.execute_notebook(nb_path, venv)
        ok = not out.startswith("[error]")
        ev = Evidence(kind=EvidenceKind.EXECUTION, source=f"executed {nb_path}",
                      content=out, success=ok, score=1.0 if ok else 0.0)
        return out, [ev]


class DocQAAgent:
    """End-to-end documentation-QA agent.

    Args:
        settings: Application settings.
        llm: GPT-OSS-120 client. If ``None``, one is constructed from settings.
        calibrator: Confidence calibrator. If ``None``, a default is used.
        offline_embeddings: When True, use the deterministic hashing embedder
            instead of Azure embeddings (handy for local smoke tests).
    """

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient | None = None,
        calibrator: ConfidenceCalibrator | None = None,
        offline_embeddings: bool = False,
    ) -> None:
        self._s = settings
        self._llm = llm or LLMClient(settings)
        self._calibrator = calibrator or ConfidenceCalibrator()
        self._offline_embeddings = offline_embeddings

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #
    def run(self, question_set: QuestionSet) -> AgentResult:
        """Answer a whole :class:`QuestionSet` and return an :class:`AgentResult`.

        Steps: prepare the repository (clone + venv + index), then answer each
        question with its own tool-use loop, then assemble the result.

        Args:
            question_set: The questions plus the ``requires_code_execution`` flag.

        Returns:
            The calibrated answers and run metadata.
        """
        notes: list[str] = []
        index = self._prepare_repo(question_set, notes)

        answers = [
            self._answer_one(q, question_set.requires_code_execution, index)
            for q in question_set.questions
        ]
        return AgentResult(
            set_id=question_set.set_id,
            answers=answers,
            repo_indexed=index is not None,
            notes=notes,
        )

    # ------------------------------------------------------------------ #
    # Repository preparation                                             #
    # ------------------------------------------------------------------ #
    def _prepare_repo(self, qs: QuestionSet, notes: list[str]) -> RepoIndex | None:
        """Clone, venv and index the repository; tolerate failures gracefully."""
        if not qs.repo_url and not qs.external_file_paths:
            return None

        workdir = Path(self._s.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        repo_path = workdir / (qs.set_id or "repo")

        venv_python: str | None = None
        if qs.repo_url:
            token = qs.gitlab_token or self._s.gitlab_token
            try:
                if not repo_path.exists():
                    clone(token, qs.repo_url, str(repo_path))
            except Exception as exc:  # noqa: BLE001
                notes.append(f"clone failed: {exc}")
                return None

            if qs.requires_code_execution:
                result = execution.create_venv(str(repo_path))
                if result.startswith("[error]"):
                    notes.append(result)
                else:
                    venv_python = result

        embedder = (
            HashingEmbedder(self._s.embedding_dim)
            if self._offline_embeddings
            else AzureEmbedder(self._llm)
        )
        try:
            return index_repository(
                str(repo_path) if qs.repo_url else str(workdir),
                embedder,
                self._s,
                extra_files=qs.external_file_paths,
                venv_python_path=venv_python,
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"indexing failed: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # Per-question answering                                             #
    # ------------------------------------------------------------------ #
    def _answer_one(
        self,
        question: str,
        allow_execution: bool,
        index: RepoIndex | None,
    ) -> QuestionAnswer:
        """Run the tool-use loop for one question and calibrate its confidence.

        Args:
            question: The question text.
            allow_execution: Whether execution tools are offered (from the
                ``requires_code_execution`` flag).
            index: The repo index, or ``None`` for conceptual questions.

        Returns:
            A fully-populated :class:`QuestionAnswer`.
        """
        tools = build_tool_schemas(allow_execution and index is not None)
        dispatcher = _ToolDispatcher(index, self._s)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(question, index)},
        ]

        payload: FinalAnswerPayload | None = None
        for _ in range(self._s.max_agent_iterations):
            resp = self._llm.chat(messages, tools=tools, tool_choice="auto")
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                # Model answered as free text; coerce into a payload.
                payload = FinalAnswerPayload(
                    answer=msg.content or "I could not determine an answer.",
                    answerable=bool(msg.content),
                    verbalized_confidence=0.3,
                    reasoning="Model returned free-form text without calling finish.",
                )
                break

            messages.append(_assistant_tool_msg(msg, tool_calls))
            finished = False
            for call in tool_calls:
                args = _safe_json(call.function.arguments)
                if call.function.name == "finish":
                    payload = _coerce_payload(args)
                    finished = True
                    result = "ok"
                else:
                    result = dispatcher.dispatch(call.function.name, args)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            if finished:
                break

        if payload is None:
            payload = FinalAnswerPayload(
                answer="Unable to complete reasoning within the step budget.",
                answerable=False,
                verbalized_confidence=0.1,
            )

        return self._finalise(question, payload, dispatcher, index)

    def _finalise(
        self,
        question: str,
        payload: FinalAnswerPayload,
        dispatcher: _ToolDispatcher,
        index: RepoIndex | None,
    ) -> QuestionAnswer:
        """Assemble evidence, run self-consistency, and calibrate confidence."""
        ev = dispatcher.evidence
        strength = evidence_strength(ev)
        grounded = payload.answerable and len(ev) > 0
        consistency = self._self_consistency(question, payload.answer, ev, index)

        signals = ConfidenceSignals(
            verbalized=payload.verbalized_confidence,
            self_consistency=consistency,
            evidence_strength=strength,
            grounded=grounded,
            answerable=payload.answerable,
        )
        confidence = self._calibrator.calibrate(signals)

        return QuestionAnswer(
            question=question,
            answer=payload.answer,
            confidence=confidence.score,
            answerable=payload.answerable,
            reasoning=payload.reasoning,
            evidence=ev,
            tool_calls=dispatcher.records,
            confidence_detail=confidence,
        )

    # ------------------------------------------------------------------ #
    # Self-consistency                                                   #
    # ------------------------------------------------------------------ #
    def _self_consistency(
        self,
        question: str,
        main_answer: str,
        evidence: list[Evidence],
        index: RepoIndex | None,
    ) -> float | None:
        """Estimate answer stability by re-answering from the gathered evidence.

        Draws several high-temperature answers conditioned on the *same*
        collected evidence and measures their embedding agreement with the main
        answer. High agreement => the answer is stable => higher confidence;
        disagreement is a strong over-confidence detector.

        Returns:
            Agreement in ``[0, 1]``, or ``None`` if disabled / not computable.
        """
        n = self._s.consistency_samples
        if n <= 0:
            return None

        context = "\n\n".join(e.content[:600] for e in evidence[:6]) or "(no evidence)"
        msgs = [
            {"role": "system", "content": "Answer the question using ONLY the context. "
                                          "Be concise (1-3 sentences)."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        try:
            resp = self._llm.chat(
                msgs, temperature=self._s.consistency_temperature, n=n
            )
            samples = [c.message.content or "" for c in resp.choices]
        except Exception:  # noqa: BLE001
            return None

        texts = [main_answer, *samples]
        embedder = index.embedder if (index and index.embedder) else None
        try:
            vectors = (
                embedder.embed(texts) if embedder is not None else self._llm.embed(texts)
            )
        except Exception:  # noqa: BLE001
            return None

        mat = _l2_normalise(np.asarray(vectors, dtype=np.float32))
        main_vec = mat[0]
        sims = mat[1:] @ main_vec  # cosine of each sample to the main answer
        agreement = float(np.clip((sims.mean() + 1.0) / 2.0, 0.0, 1.0))
        return agreement

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _user_prompt(self, question: str, index: RepoIndex | None) -> str:
        """Build the per-question user message, including a repo hint."""
        if index is None:
            return f"Question: {question}\n\n(No repository is available; answer from knowledge.)"
        return (
            f"Repository is indexed at: {index.repo_path}\n"
            f"Use the tools to gather evidence, then call `finish`.\n\n"
            f"Question: {question}"
        )


# --------------------------------------------------------------------------- #
# Module-level helpers                                                         #
# --------------------------------------------------------------------------- #
def _safe_json(raw: str | None) -> dict[str, Any]:
    """Parse a tool-call argument string, tolerating malformed JSON."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _coerce_payload(args: dict[str, Any]) -> FinalAnswerPayload:
    """Validate the model's ``finish`` arguments into a :class:`FinalAnswerPayload`."""
    try:
        return FinalAnswerPayload.model_validate(args)
    except Exception:  # noqa: BLE001
        return FinalAnswerPayload(
            answer=str(args.get("answer", "No answer produced.")),
            answerable=bool(args.get("answerable", True)),
            verbalized_confidence=float(args.get("verbalized_confidence", 0.3)),
            reasoning=str(args.get("reasoning", "")),
        )


def _assistant_tool_msg(msg: Any, tool_calls: Any) -> dict[str, Any]:
    """Reconstruct the assistant message (with tool_calls) for the transcript."""
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in tool_calls
        ],
    }


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation for cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
