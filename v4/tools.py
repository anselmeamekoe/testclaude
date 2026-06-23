"""The agent's tool layer.

This module bridges the LLM's function-calling interface and two concrete
capabilities:

1. **Retrieval** over the indexed repo (``search_code``, ``search_docs``,
   ``read_file``, ``list_files``) — always available.
2. **Execution** (``execute_python_snippet``, ``execute_python_file``,
   ``execute_notebook``, ``install_packages``) — enabled **only** when the
   leaderboard payload sets ``code_execution=True``.

The execution functions are the ones the organisers already provide (in your
``tools/`` package). Rather than hard-code import paths, the toolbox receives an
:class:`ExecutionBackend` whose method signatures match the provided functions.
Wire your real implementation via :func:`build_default_backend`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Protocol

from sparrow_agent.models import ToolInvocation, Trajectory
from sparrow_agent.rag import Retriever

# A terminal "tool" the model calls to emit its final structured answer.
SUBMIT_ANSWER = "submit_answer"


class ExecutionBackend(Protocol):
    """Interface matching the organisers' execution helpers.

    Implementations are expected to be the functions you already have under
    ``tools/code`` and ``tools/gitlab``. Signatures mirror those exactly.
    """

    def execute_python_snippet(
        self, code: str, venv_python_path: str | None = None, env: dict[str, str] | None = None
    ) -> str: ...

    def execute_python_file(
        self, file_path: str, venv_python_path: str | None = None, env: dict[str, str] | None = None
    ) -> str: ...

    def execute_notebook(
        self,
        notebook_path: str,
        venv_python_path: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str: ...

    def install_packages(self, packages: list[str], venv_python_path: str) -> str: ...


class _CallableBackend:
    """Adapts a set of plain callables into an :class:`ExecutionBackend`.

    Lets you inject the organisers' functions without subclassing anything.
    """

    def __init__(
        self,
        execute_python_snippet: Callable[..., str],
        execute_python_file: Callable[..., str],
        execute_notebook: Callable[..., str],
        install_packages: Callable[..., str],
    ) -> None:
        """Store the provided callables.

        Args:
            execute_python_snippet: The organisers' snippet runner.
            execute_python_file: The organisers' file runner.
            execute_notebook: The organisers' notebook runner.
            install_packages: The organisers' package installer.
        """
        self.execute_python_snippet = execute_python_snippet
        self.execute_python_file = execute_python_file
        self.execute_notebook = execute_notebook
        self.install_packages = install_packages


def build_default_backend() -> ExecutionBackend:
    """Import the organisers' execution functions from the ``tools`` package.

    Adjust the import paths here to match your repository layout. Per your tree:

        tools/code/code_execution.py  -> execute_* functions
        tools/code/env_setup.py       -> install_packages

    Returns:
        An :class:`ExecutionBackend` bound to your real functions.

    Raises:
        RuntimeError: If the ``tools`` package cannot be imported, with a hint.
    """
    try:
        from tools.code.code_execution import (  # type: ignore
            execute_notebook,
            execute_python_file,
            execute_python_snippet,
        )
        from tools.code.env_setup import install_packages  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on host project
        raise RuntimeError(
            "Could not import the organisers' execution tools. Edit "
            "build_default_backend() in sparrow_agent/tools.py to point at your "
            "actual modules, or pass a custom ExecutionBackend to ToolBox."
        ) from exc

    return _CallableBackend(
        execute_python_snippet=execute_python_snippet,
        execute_python_file=execute_python_file,
        execute_notebook=execute_notebook,
        install_packages=install_packages,
    )


class ToolBox:
    """Exposes tools to the LLM and dispatches the model's tool calls.

    A new :class:`ToolBox` is created per question-set run, because it carries
    per-run state: the repo path, the venv path, and whether execution is on.
    """

    def __init__(
        self,
        retriever: Retriever,
        repo_root: str,
        code_execution: bool,
        backend: ExecutionBackend | None = None,
        venv_python_path: str | None = None,
        exec_env: dict[str, str] | None = None,
        result_char_limit: int = 6000,
    ) -> None:
        """Initialise the toolbox.

        Args:
            retriever: Built retriever for the cloned repo.
            repo_root: Absolute path to the cloned repo (for file reads).
            code_execution: Whether execution tools are exposed at all.
            backend: Execution backend; required when ``code_execution`` is True.
            venv_python_path: Path to the project venv's python interpreter.
            exec_env: Extra environment variables (tokens, keys) for execution.
            result_char_limit: Hard cap on tool output length fed back to the LLM.
        """
        self._retriever = retriever
        self._repo_root = Path(repo_root)
        self._code_execution = code_execution
        self._backend = backend
        self._venv_python_path = venv_python_path
        self._exec_env = exec_env or {}
        self._limit = result_char_limit

    # ---- schema --------------------------------------------------------------

    def schemas(self) -> list[dict]:
        """Return the JSON schemas of every *enabled* tool, OpenAI format.

        Execution tools are included only when ``code_execution`` is True, which
        is how the agent is steered away from running code when the question set
        does not require it.

        Returns:
            A list of ``{"type": "function", "function": {...}}`` schemas.
        """
        tools: list[dict] = [
            _fn(
                "search_code",
                "Semantic search over the repository's SOURCE CODE. Use to find "
                "where something is defined, how a function behaves, defaults, etc.",
                {"query": _str("What to look for, phrased as a description or symbol."),
                 "k": _int("Number of snippets to return (default 6).")},
                ["query"],
            ),
            _fn(
                "search_docs",
                "Semantic search over DOCUMENTATION (README, .md, .rst, docstrings "
                "captured as prose). Use for conceptual/usage questions.",
                {"query": _str("What to look for."),
                 "k": _int("Number of snippets to return (default 6).")},
                ["query"],
            ),
            _fn(
                "read_file",
                "Read an exact slice of a repository file by path and line range. "
                "Use after a search to confirm details verbatim.",
                {"path": _str("Repository-relative path, e.g. 'src/app/config.py'."),
                 "start_line": _int("1-based first line (default 1)."),
                 "end_line": _int("1-based last line (default: 200 lines after start).")},
                ["path"],
            ),
            _fn(
                "list_files",
                "List repository files, optionally under a subdirectory. Use to "
                "orient yourself before searching.",
                {"subdir": _str("Optional subdirectory to list (default: repo root)."),
                 "pattern": _str("Optional substring filter on the path.")},
                [],
            ),
        ]

        if self._code_execution:
            tools += [
                _fn(
                    "execute_python_snippet",
                    "Run a short Python snippet inside the project's venv and return "
                    "stdout/stderr/exit code. Use to VERIFY a value empirically "
                    "(e.g. read a config, compute a default, inspect an object).",
                    {"code": _str("Python source to run. Print what you want to observe.")},
                    ["code"],
                ),
                _fn(
                    "execute_python_file",
                    "Run an existing repository .py file inside the project's venv.",
                    {"file_path": _str("Repo-relative or absolute path to the .py file.")},
                    ["file_path"],
                ),
                _fn(
                    "execute_notebook",
                    "Execute a Jupyter notebook and return each cell's code and output.",
                    {"notebook_path": _str("Repo-relative or absolute path to the .ipynb.")},
                    ["notebook_path"],
                ),
                _fn(
                    "install_packages",
                    "Install PyPI packages into the project's venv. Use only when an "
                    "execution fails with ModuleNotFoundError.",
                    {"packages": {"type": "array", "items": {"type": "string"},
                                  "description": "Package specs, e.g. ['numpy>=1.26']."}},
                    ["packages"],
                ),
            ]

        tools.append(
            _fn(
                SUBMIT_ANSWER,
                "Emit the FINAL answer for this question and end the turn. Always "
                "call this exactly once when you are done. Prefer not_known=true "
                "over guessing: overconfident wrong answers are penalised.",
                {
                    "answer": _str("The answer text, or what is missing if not_known."),
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"],
                                   "description": "Your honest confidence given the evidence you gathered."},
                    "evidence": {"type": "array",
                                 "items": {"type": "string", "enum": ["files", "execution"]},
                                 "description": "Which sources actually supported the answer."},
                    "not_known": {"type": "boolean",
                                  "description": "True if the repo/files do not contain the answer."},
                    "reasoning": _str("One or two sentences citing the specific files/lines or "
                                      "execution output that justify the answer."),
                },
                ["answer", "confidence", "not_known"],
            )
        )
        return tools

    # ---- dispatch ------------------------------------------------------------

    def dispatch(self, name: str, arguments: dict, trajectory: Trajectory) -> str:
        """Execute a tool call by name and record it on the trajectory.

        Args:
            name: Tool name chosen by the model.
            arguments: Parsed JSON arguments.
            trajectory: The running trace; updated in place with telemetry.

        Returns:
            A string to feed back to the model as the tool result.
        """
        try:
            result = self._run(name, arguments, trajectory)
            ok = True
        except Exception as exc:  # surface errors to the model rather than crash
            result = f"[tool error] {type(exc).__name__}: {exc}"
            ok = False

        result = _truncate(result, self._limit)
        trajectory.steps.append(
            ToolInvocation(name=name, arguments=arguments, ok=ok, result_preview=result[:300])
        )
        trajectory.num_steps = len(trajectory.steps)
        return result

    def _run(self, name: str, args: dict, trajectory: Trajectory) -> str:
        """Internal dispatcher mapping a tool name to its implementation.

        Args:
            name: Tool name.
            args: Tool arguments.
            trajectory: Running trace (updated with retrieval/execution flags).

        Returns:
            The tool's textual result.

        Raises:
            ValueError: If ``name`` is unknown or disabled.
        """
        if name == "search_code":
            return self._do_search(args, trajectory, kind="code")
        if name == "search_docs":
            return self._do_search(args, trajectory, kind="doc")
        if name == "read_file":
            trajectory.used_files = True
            return self._read_file(args)
        if name == "list_files":
            return self._list_files(args)

        if name in {"execute_python_snippet", "execute_python_file",
                    "execute_notebook", "install_packages"}:
            if not self._code_execution or self._backend is None:
                raise ValueError("Execution tools are disabled for this question set.")
            return self._do_execute(name, args, trajectory)

        raise ValueError(f"Unknown tool: {name}")

    # ---- retrieval tools -----------------------------------------------------

    def _do_search(self, args: dict, trajectory: Trajectory, kind: str) -> str:
        """Run a code/doc search and format the hits for the model.

        Args:
            args: ``{"query": str, "k": int?}``.
            trajectory: Updated with ``used_files`` and ``max_retrieval_score``.
            kind: ``"code"`` or ``"doc"``.

        Returns:
            A readable, line-referenced rendering of the hits.
        """
        query = str(args.get("query", "")).strip()
        k = int(args.get("k") or 0) or None
        hits = (self._retriever.search_code if kind == "code" else self._retriever.search_docs)(
            query, k
        )
        if hits:
            trajectory.used_files = True
            trajectory.max_retrieval_score = max(
                trajectory.max_retrieval_score, max(h.score for h in hits)
            )
        if not hits:
            return f"No {kind} matches for: {query!r}"
        blocks = []
        for hit in hits:
            c = hit.chunk
            blocks.append(
                f"[{c.path}:{c.start_line}-{c.end_line}] (similarity={hit.score:.2f})\n{c.text}"
            )
        return "\n\n---\n\n".join(blocks)

    def _read_file(self, args: dict) -> str:
        """Return an exact line slice of a repository file.

        Args:
            args: ``{"path": str, "start_line": int?, "end_line": int?}``.

        Returns:
            The requested lines, prefixed with 1-based line numbers.
        """
        rel = str(args.get("path", "")).lstrip("/")
        target = (self._repo_root / rel).resolve()
        if not str(target).startswith(str(self._repo_root.resolve())):
            return "[tool error] Path escapes the repository root."
        if not target.is_file():
            return f"[tool error] File not found: {rel}"
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        start = max(int(args.get("start_line") or 1), 1)
        end = int(args.get("end_line") or (start + 199))
        end = min(end, len(lines))
        body = "\n".join(f"{i:>5}\t{lines[i - 1]}" for i in range(start, end + 1))
        return f"{rel} (lines {start}-{end} of {len(lines)}):\n{body}"

    def _list_files(self, args: dict) -> str:
        """List repository files, optionally filtered.

        Args:
            args: ``{"subdir": str?, "pattern": str?}``.

        Returns:
            A newline-separated list of relative paths (capped to 300 entries).
        """
        subdir = str(args.get("subdir", "")).lstrip("/")
        pattern = str(args.get("pattern", "")).lower()
        base = (self._repo_root / subdir).resolve()
        if not str(base).startswith(str(self._repo_root.resolve())) or not base.exists():
            return f"[tool error] Invalid subdir: {subdir}"
        results: list[str] = []
        skip = {".git", ".venv", "node_modules", "__pycache__"}
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in skip]
            for name in filenames:
                rel = str((Path(dirpath) / name).relative_to(self._repo_root))
                if not pattern or pattern in rel.lower():
                    results.append(rel)
        results.sort()
        listing = "\n".join(results[:300])
        more = "" if len(results) <= 300 else f"\n... (+{len(results) - 300} more)"
        return listing + more

    # ---- execution tools -----------------------------------------------------

    def _do_execute(self, name: str, args: dict, trajectory: Trajectory) -> str:
        """Run an execution tool through the injected backend.

        Args:
            name: One of the execution tool names.
            args: Tool arguments.
            trajectory: Updated with ``used_execution`` on success.

        Returns:
            The backend's textual output.
        """
        backend = self._backend
        assert backend is not None  # guarded by caller
        env = self._exec_env or None

        if name == "execute_python_snippet":
            out = backend.execute_python_snippet(
                code=str(args["code"]), venv_python_path=self._venv_python_path, env=env
            )
        elif name == "execute_python_file":
            out = backend.execute_python_file(
                file_path=self._abs(args["file_path"]),
                venv_python_path=self._venv_python_path,
                env=env,
            )
        elif name == "execute_notebook":
            out = backend.execute_notebook(
                notebook_path=self._abs(args["notebook_path"]),
                venv_python_path=self._venv_python_path,
                env=env,
            )
        elif name == "install_packages":
            pkgs = args.get("packages") or []
            if isinstance(pkgs, str):
                pkgs = [pkgs]
            out = backend.install_packages(
                packages=list(pkgs), venv_python_path=self._venv_python_path or ""
            )
        else:  # pragma: no cover - guarded earlier
            raise ValueError(name)

        trajectory.used_execution = True
        return out

    def _abs(self, path: str) -> str:
        """Resolve a possibly-relative repo path to an absolute path.

        Args:
            path: Relative or absolute file path.

        Returns:
            An absolute path string.
        """
        p = Path(path)
        return str(p if p.is_absolute() else (self._repo_root / path))


# ---------------------------------------------------------------------------
# Small JSON-schema helpers (keep the schemas() method readable).
# ---------------------------------------------------------------------------


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Build one OpenAI function-tool schema.

    Args:
        name: Function name.
        description: When-to-use guidance for the model.
        properties: JSON-schema properties for the arguments.
        required: Names of required arguments.

    Returns:
        A function-tool schema dict.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _str(description: str) -> dict:
    """A string-typed JSON-schema property."""
    return {"type": "string", "description": description}


def _int(description: str) -> dict:
    """An integer-typed JSON-schema property."""
    return {"type": "integer", "description": description}


def _truncate(text: str, limit: int) -> str:
    """Truncate tool output so a single result cannot blow the context window.

    Args:
        text: Raw tool output.
        limit: Maximum characters to keep.

    Returns:
        Possibly-truncated text with a marker.
    """
    if len(text) <= limit:
        return text
    head = text[: limit - 200]
    return f"{head}\n... [truncated {len(text) - len(head)} chars]"


def parse_tool_arguments(raw: str) -> dict:
    """Parse the JSON argument string from a tool call, defensively.

    Args:
        raw: The ``function.arguments`` string from the model.

    Returns:
        Parsed arguments, or ``{}`` on failure.
    """
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
