"""End-to-end orchestration.

:class:`AgentPipeline` turns a leaderboard :class:`Input` into an :class:`Output`:

1. Clone the GitLab repo (token-authenticated).
2. If ``code_execution`` is True, create the project's virtual environment.
3. Chunk + embed + index the repo (FAISS RAG).
4. For each question, run the agent and collect a calibrated answer.

Provisioning (clone/venv) and execution both delegate to the organisers'
``tools/`` package. Import paths are wired in :func:`_build_provisioner` and
:func:`sparrow_agent.tools.build_default_backend`; adjust them to your tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from sparrow_agent.agent import DocQAAgent
from sparrow_agent.calibration import CalibrationEngine
from sparrow_agent.config import Settings, get_settings
from sparrow_agent.embeddings import EmbeddingClient
from sparrow_agent.llm import LLMClient
from sparrow_agent.models import AnswerItem, Input, Output
from sparrow_agent.rag import Retriever
from sparrow_agent.tools import ExecutionBackend, ToolBox, build_default_backend


class _Provisioner:
    """Wraps the organisers' ``clone`` and ``create_venv`` helpers."""

    def __init__(
        self,
        clone: Callable[[str, str, str], None],
        create_venv: Callable[[str], str],
    ) -> None:
        """Store the provisioning callables.

        Args:
            clone: ``clone(gitlab_token, repo_url, dest_folder) -> None``.
            create_venv: ``create_venv(project_path) -> str`` (returns python path).
        """
        self.clone = clone
        self.create_venv = create_venv


def _build_provisioner() -> _Provisioner:
    """Import the organisers' clone/venv helpers from the ``tools`` package.

    Returns:
        A :class:`_Provisioner` bound to your real functions.

    Raises:
        RuntimeError: If the helpers cannot be imported, with a hint.
    """
    try:
        from tools.code.env_setup import create_venv  # type: ignore
        from tools.gitlab.clone import clone  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on host project
        raise RuntimeError(
            "Could not import clone/create_venv from your tools package. Edit "
            "_build_provisioner() in sparrow_agent/pipeline.py to match your layout."
        ) from exc
    return _Provisioner(clone=clone, create_venv=create_venv)


class AgentPipeline:
    """Runs the whole agent over one leaderboard payload."""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        embedder: EmbeddingClient,
        agent: DocQAAgent,
        provisioner: _Provisioner,
        backend: ExecutionBackend | None,
    ) -> None:
        """Construct the pipeline from its collaborators.

        Prefer :meth:`from_settings` unless you need to inject custom backends
        (e.g. in tests).

        Args:
            settings: Global configuration.
            llm: Chat client.
            embedder: Embedding client.
            agent: The reasoning agent.
            provisioner: Clone/venv helper bundle.
            backend: Execution backend (may be None if execution is never used).
        """
        self._settings = settings
        self._llm = llm
        self._embedder = embedder
        self._agent = agent
        self._provisioner = provisioner
        self._backend = backend

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "AgentPipeline":
        """Build a fully-wired pipeline from configuration.

        Args:
            settings: Optional settings override; defaults to :func:`get_settings`.

        Returns:
            A ready-to-run :class:`AgentPipeline`.
        """
        settings = settings or get_settings()
        llm = LLMClient(settings)
        embedder = EmbeddingClient(settings)
        calibration = CalibrationEngine(settings, llm)
        agent = DocQAAgent(settings, llm, embedder, calibration)
        provisioner = _build_provisioner()
        backend: ExecutionBackend | None
        try:
            backend = build_default_backend()
        except RuntimeError:
            backend = None  # execution simply stays unavailable
        return cls(settings, llm, embedder, agent, provisioner, backend)

    def run(self, payload: Input) -> Output:
        """Execute the full flow for one submission.

        Args:
            payload: The leaderboard input payload.

        Returns:
            An :class:`Output` echoing ``submission_id`` with one answer per
            question, in input order.
        """
        repo_root = self._clone_repo(payload)
        venv_python = self._maybe_create_venv(payload, repo_root)

        retriever = Retriever(self._settings, self._embedder)
        retriever.build(repo_root)

        toolbox = ToolBox(
            retriever=retriever,
            repo_root=repo_root,
            code_execution=payload.code_execution,
            backend=self._backend if payload.code_execution else None,
            venv_python_path=venv_python,
            exec_env=self._exec_env(payload),
        )

        answers: list[AnswerItem] = []
        for question in payload.template:
            answers.append(self._agent.answer(question, toolbox, payload.template_title))

        return Output(submission_id=payload.submission_id, answers=answers)

    # ---- steps ---------------------------------------------------------------

    def _clone_repo(self, payload: Input) -> str:
        """Clone the repo into a per-submission workspace folder.

        Args:
            payload: The input payload (repo URL + GitLab token).

        Returns:
            Absolute path to the cloned repository.
        """
        dest = Path(self._settings.workspace_dir) / payload.submission_id / "repo"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            self._provisioner.clone(payload.token_gitlab, payload.repo_url, str(dest))
        return str(dest)

    def _maybe_create_venv(self, payload: Input, repo_root: str) -> str | None:
        """Create the project venv only when execution is enabled.

        Args:
            payload: The input payload (its ``code_execution`` flag gates this).
            repo_root: Path to the cloned repository.

        Returns:
            The venv python path, or ``None`` when execution is disabled or venv
            creation fails (the agent then relies on retrieval only).
        """
        if not payload.code_execution:
            return None
        try:
            venv_python = self._provisioner.create_venv(repo_root)
        except Exception:
            return None
        if isinstance(venv_python, str) and venv_python.startswith("[error]"):
            return None
        return venv_python

    @staticmethod
    def _exec_env(payload: Input) -> dict[str, str]:
        """Build the environment variables exposed to executed code.

        Surfaces the team tokens/keys so notebooks and scripts that need to reach
        Sparrow or the dataset can authenticate.

        Args:
            payload: The input payload carrying the secrets.

        Returns:
            A dict of environment variables for the execution backend.
        """
        return {
            "SPARROW_TOKEN": payload.token_sparrow,
            "GITLAB_TOKEN": payload.token_gitlab,
            "DATASET_ACCESS_KEY": payload.access_key,
            "DATASET_SECRET_KEY": payload.secret_key,
        }


def main() -> None:
    """CLI entry point: read an Input JSON, run the pipeline, print Output JSON.

    Usage:
        poetry run sparrow-agent --input payload.json
        cat payload.json | poetry run sparrow-agent
    """
    parser = argparse.ArgumentParser(description="Run the Sparrow documentation-QA agent.")
    parser.add_argument(
        "--input", type=str, default=None, help="Path to an Input JSON file (else read stdin)."
    )
    parser.add_argument("--output", type=str, default=None, help="Where to write Output JSON.")
    args = parser.parse_args()

    raw = Path(args.input).read_text() if args.input else sys.stdin.read()
    payload = Input.model_validate_json(raw)

    pipeline = AgentPipeline.from_settings()
    output = pipeline.run(payload)

    serialized = output.model_dump_json(indent=2)
    if args.output:
        Path(args.output).write_text(serialized)
    else:
        print(serialized)


if __name__ == "__main__":
    main()
