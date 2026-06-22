"""Execution tools: the organizer-provided runtime, wrapped for the agent.

The hackathon organizers supply the real execution sandbox (clone GitLab repo,
create venv, install packages, run files/notebooks, execute snippets). This module
defines a narrow :class:`ExecutionBackend` protocol describing exactly those
capabilities, plus:

* :class:`LocalExecutionBackend` — a working reference implementation using
  ``git``, ``venv`` and ``subprocess`` so the whole agent is runnable standalone
  and in CI. In the competition you swap this for an adapter around the organizer
  API while keeping the same interface.
* Factory functions that wrap each backend capability as an agent :class:`Tool`
  with a proper input schema and evidence reporting.

Separating the *interface* from the *implementation* is what lets the agent's
reasoning stay identical whether it runs locally or against the organizer sandbox.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional, Protocol

from ..models import Evidence, ExecutionResult, SourceType
from .base import Tool, ToolResult

# Output captured from subprocesses is truncated to this many characters before
# being shown to the model, to protect the context window from huge logs.
_MAX_OUTPUT_CHARS = 4000


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Trim ``text`` to ``limit`` characters, marking that truncation occurred.

    Args:
        text: Text to bound.
        limit: Maximum characters to keep.

    Returns:
        The original text, or a head+tail-trimmed version with a marker.
    """
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... [truncated {len(text) - limit} chars] ...\n{tail}"


class ExecutionBackend(Protocol):
    """The execution capabilities the organizers expose (and we depend on).

    Implement this protocol to bridge to the real sandbox. Every method returns an
    :class:`ExecutionResult` (except :meth:`clone_repo`, which returns a path) so
    the agent gets a uniform, typed view of runtime behaviour.
    """

    def clone_repo(self, repo_url: str) -> str:
        """Clone ``repo_url`` and return the local path it was cloned to."""
        ...

    def create_venv(self, repo_path: str) -> ExecutionResult:
        """Create a virtual environment for the project at ``repo_path``."""
        ...

    def install_packages(self, repo_path: str, packages: list[str]) -> ExecutionResult:
        """Install ``packages`` (or the project's own deps) into the venv."""
        ...

    def run_python_file(
        self, repo_path: str, file_path: str, args: list[str]
    ) -> ExecutionResult:
        """Run ``file_path`` (relative to ``repo_path``) with optional ``args``."""
        ...

    def run_notebook(self, repo_path: str, notebook_path: str) -> ExecutionResult:
        """Execute a Jupyter notebook end-to-end and capture its output."""
        ...

    def execute_snippet(self, repo_path: str, code: str) -> ExecutionResult:
        """Execute an ad-hoc Python ``code`` snippet inside the project context."""
        ...


class LocalExecutionBackend:
    """Reference :class:`ExecutionBackend` using local ``git``/``venv``/``subprocess``.

    Intended for development, tests, and as a template for the organizer adapter.
    It enforces per-command timeouts and captures stdout/stderr so failures surface
    as data rather than hangs.
    """

    def __init__(self, workdir: str, timeout_seconds: int = 120) -> None:
        """Prepare the working directory.

        Args:
            workdir: Base directory under which repos and venvs are created.
            timeout_seconds: Max wall-clock seconds per executed command.
        """
        self.workdir = os.path.abspath(workdir)
        self.timeout_seconds = timeout_seconds
        os.makedirs(self.workdir, exist_ok=True)

    def _venv_python(self, repo_path: str) -> str:
        """Return the venv's Python interpreter path, or the system one if absent.

        Args:
            repo_path: Project root that may contain a ``.venv``.

        Returns:
            Absolute path to the interpreter to use for execution.
        """
        candidate = os.path.join(repo_path, ".venv", "bin", "python")
        return candidate if os.path.exists(candidate) else sys.executable

    def _run(self, cmd: list[str], cwd: str) -> ExecutionResult:
        """Run a subprocess with timeout and capture, as an :class:`ExecutionResult`.

        Args:
            cmd: Argument vector to execute.
            cwd: Working directory for the process.

        Returns:
            A populated :class:`ExecutionResult` (``success`` reflects exit code 0).
        """
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            return ExecutionResult(
                success=proc.returncode == 0,
                return_code=proc.returncode,
                stdout=_truncate(proc.stdout),
                stderr=_truncate(proc.stderr),
                duration_seconds=round(time.time() - start, 3),
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                return_code=None,
                stderr=f"Timed out after {self.timeout_seconds}s",
                duration_seconds=float(self.timeout_seconds),
            )

    def clone_repo(self, repo_url: str) -> str:
        """Shallow-clone ``repo_url`` under the working directory.

        Args:
            repo_url: Git/GitLab URL to clone.

        Returns:
            Local filesystem path of the clone.
        """
        name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git") or "repo"
        dest = os.path.join(self.workdir, name)
        if not os.path.exists(dest):
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, dest],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        return dest

    def create_venv(self, repo_path: str) -> ExecutionResult:
        """Create ``.venv`` inside ``repo_path`` using the stdlib ``venv`` module."""
        return self._run([sys.executable, "-m", "venv", ".venv"], cwd=repo_path)

    def install_packages(self, repo_path: str, packages: list[str]) -> ExecutionResult:
        """Pip-install explicit ``packages``; if none given, try ``requirements.txt``.

        Args:
            repo_path: Project root (must contain a ``.venv``).
            packages: Explicit package specs; empty to install from requirements.

        Returns:
            The pip command's :class:`ExecutionResult`.
        """
        py = self._venv_python(repo_path)
        if packages:
            cmd = [py, "-m", "pip", "install", *packages]
        elif os.path.exists(os.path.join(repo_path, "requirements.txt")):
            cmd = [py, "-m", "pip", "install", "-r", "requirements.txt"]
        else:
            return ExecutionResult(success=True, stdout="No packages specified; nothing to install.")
        return self._run(cmd, cwd=repo_path)

    def run_python_file(
        self, repo_path: str, file_path: str, args: Optional[list[str]] = None
    ) -> ExecutionResult:
        """Execute a Python file inside the project venv."""
        py = self._venv_python(repo_path)
        return self._run([py, file_path, *(args or [])], cwd=repo_path)

    def run_notebook(self, repo_path: str, notebook_path: str) -> ExecutionResult:
        """Execute a notebook via ``jupyter nbconvert --execute``.

        Requires ``jupyter`` in the venv; otherwise the failure (with stderr) is
        returned so the agent can react and lower confidence.
        """
        py = self._venv_python(repo_path)
        return self._run(
            [py, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--stdout", notebook_path],
            cwd=repo_path,
        )

    def execute_snippet(self, repo_path: str, code: str) -> ExecutionResult:
        """Run an ad-hoc snippet with the project root on ``sys.path``.

        Args:
            repo_path: Project root, prepended to ``PYTHONPATH`` so imports resolve.
            code: Python source to execute via ``python -c``.

        Returns:
            The snippet's :class:`ExecutionResult`.
        """
        py = self._venv_python(repo_path)
        return self._run([py, "-c", code], cwd=repo_path)


def _execution_result_to_tool_result(label: str, result: ExecutionResult) -> ToolResult:
    """Adapt a raw :class:`ExecutionResult` into a :class:`ToolResult` for the agent.

    Produces an EXECUTION-typed :class:`Evidence` item so the calibrator can credit
    (or penalize) answers grounded in actual runtime output.

    Args:
        label: Short tag identifying what was run (e.g. ``"run_python_file:foo.py"``).
        result: The execution outcome.

    Returns:
        A :class:`ToolResult` summarizing the run, carrying one evidence item.
    """
    status = "SUCCESS" if result.success else "FAILURE"
    body = (
        f"[{status}] {label}\n"
        f"return_code={result.return_code} duration={result.duration_seconds}s\n"
        f"--- stdout ---\n{result.stdout or '(empty)'}\n"
        f"--- stderr ---\n{result.stderr or '(empty)'}"
    )
    evidence = Evidence(
        source_type=SourceType.EXECUTION,
        reference=label,
        snippet=_truncate(result.stdout or result.stderr, 600),
        score=1.0 if result.success else 0.2,
    )
    return ToolResult(
        content=body,
        evidence=[evidence],
        success=result.success,
        metadata={"execution_result": result.model_dump()},
    )


def build_execution_tools(backend: ExecutionBackend, repo_path_getter) -> list[Tool]:
    """Wrap an :class:`ExecutionBackend` as agent-callable :class:`Tool` objects.

    The repo path is resolved lazily through ``repo_path_getter`` so the same tool
    objects work before and after the repo is cloned within a run.

    Args:
        backend: The execution backend to expose.
        repo_path_getter: Zero-arg callable returning the current repo path
            (raising or returning ``None`` if no repo is available yet).

    Returns:
        A list of execution :class:`Tool` objects: setup, run file, run notebook,
        execute snippet.
    """

    def _repo() -> str:
        """Resolve the current repo path or raise a clear error for the agent."""
        path = repo_path_getter()
        if not path:
            raise RuntimeError("No repository is available; cannot execute code.")
        return path

    def setup_environment(packages: Optional[list[str]] = None) -> ToolResult:
        """Create the venv and install dependencies in one step.

        Args:
            packages: Optional explicit packages; falls back to ``requirements.txt``.
        """
        repo = _repo()
        venv = backend.create_venv(repo)
        if not venv.success:
            return _execution_result_to_tool_result("create_venv", venv)
        install = backend.install_packages(repo, packages or [])
        return _execution_result_to_tool_result(
            f"setup_environment(packages={packages or 'requirements.txt'})", install
        )

    def run_python_file(file_path: str, args: Optional[list[str]] = None) -> ToolResult:
        """Run a Python file from the repo and capture its output."""
        result = backend.run_python_file(_repo(), file_path, args or [])
        return _execution_result_to_tool_result(f"run_python_file:{file_path}", result)

    def run_notebook(notebook_path: str) -> ToolResult:
        """Execute a notebook from the repo and capture its output."""
        result = backend.run_notebook(_repo(), notebook_path)
        return _execution_result_to_tool_result(f"run_notebook:{notebook_path}", result)

    def execute_code(code: str) -> ToolResult:
        """Execute an ad-hoc Python snippet against the repo to verify behaviour."""
        result = backend.execute_snippet(_repo(), code)
        return _execution_result_to_tool_result("execute_code", result)

    return [
        Tool(
            name="setup_environment",
            description=(
                "Create a virtual environment and install dependencies for the "
                "cloned repo. Call this BEFORE running files/notebooks/snippets "
                "that import third-party packages. Optionally pass explicit "
                "'packages'; otherwise requirements.txt is used."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional explicit package specs to install.",
                    }
                },
            },
            handler=setup_environment,
        ),
        Tool(
            name="run_python_file",
            description=(
                "Run an existing Python file from the repository and capture "
                "stdout/stderr. Use to observe real runtime behaviour, outputs, "
                "or errors that cannot be determined by reading code alone."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Repo-relative path to the .py file to run.",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional command-line arguments.",
                    },
                },
                "required": ["file_path"],
            },
            handler=run_python_file,
        ),
        Tool(
            name="run_notebook",
            description=(
                "Execute a Jupyter notebook end-to-end and capture its output. "
                "Use when the answer depends on notebook results."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "notebook_path": {
                        "type": "string",
                        "description": "Repo-relative path to the .ipynb file.",
                    }
                },
                "required": ["notebook_path"],
            },
            handler=run_notebook,
        ),
        Tool(
            name="execute_code",
            description=(
                "Execute a short ad-hoc Python snippet inside the repo context "
                "(repo root on sys.path). Ideal for calling a function with sample "
                "inputs, checking a return value, or confirming a behaviour you "
                "could not determine by reading code."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source to execute. Print what you need.",
                    }
                },
                "required": ["code"],
            },
            handler=execute_code,
        ),
    ]
