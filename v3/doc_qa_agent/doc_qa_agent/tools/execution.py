"""Code-execution tools.

These mirror the signatures the organizers provide (``create_venv``,
``install_packages``, ``execute_python_file``, ``execute_python_snippet``,
``execute_notebook``). If the organizers ship their own implementations, swap
this module out — the agent only depends on the signatures below. They are
implemented here so the project runs standalone.

All execution is sandboxed to a subprocess and never raises on user-code
errors: failures are captured and returned as text so the agent can *reason*
about them (e.g. install a missing dependency and retry) instead of crashing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path


def create_venv(project_path: str) -> str:
    """Create (or reuse) a virtual environment for a project and install deps.

    Idempotent: if a ``.venv`` already exists it is reused. Dependencies are
    installed from ``requirements.txt`` and/or ``pyproject.toml`` when present.

    Args:
        project_path: Absolute path to the repository root.

    Returns:
        Absolute path to the venv Python executable, or a string starting with
        ``"[error]"`` when creation fails.
    """
    root = Path(project_path)
    if not root.is_dir():
        return f"[error] project_path does not exist: {project_path}"

    venv_dir = root / ".venv"
    py = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    try:
        if not py.exists():
            venv.create(venv_dir, with_pip=True)
        # Best-effort dependency install; failures are non-fatal.
        req = root / "requirements.txt"
        if req.exists():
            subprocess.run(  # noqa: S603
                [str(py), "-m", "pip", "install", "-q", "-r", str(req)],
                capture_output=True, text=True, check=False,
            )
        if (root / "pyproject.toml").exists():
            subprocess.run(  # noqa: S603
                [str(py), "-m", "pip", "install", "-q", "-e", "."],
                cwd=str(root), capture_output=True, text=True, check=False,
            )
        return str(py)
    except Exception as exc:  # noqa: BLE001
        return f"[error] venv creation failed: {exc}"


def install_packages(packages: list[str], venv_python_path: str) -> str:
    """Install one or more PyPI packages into a virtual environment.

    Args:
        packages: pip-style specs, e.g. ``["numpy>=1.26", "torch==2.3.0"]``.
        venv_python_path: Path to the venv Python executable.

    Returns:
        ``"Successfully installed: ..."`` on success, else ``"[error] ..."``.
    """
    if not packages:
        return "[error] No packages specified."
    if not Path(venv_python_path).exists():
        return f"[error] venv python not found: {venv_python_path}"
    result = subprocess.run(  # noqa: S603
        [venv_python_path, "-m", "pip", "install", *packages],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return f"[error] pip install failed.\n{result.stdout}\n{result.stderr}"
    return f"Successfully installed: {', '.join(packages)}"


def _merged_env(env: dict[str, str] | None) -> dict[str, str]:
    """Merge extra variables on top of the current process environment."""
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return merged


def execute_python_file(
    file_path: str,
    venv_python_path: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run a Python ``.py`` file and return stdout, stderr and exit code.

    Args:
        file_path: Absolute path to the ``.py`` file to run.
        venv_python_path: Python executable to use; falls back to the current
            interpreter when omitted.
        env: Extra environment variables merged over the current environment.

    Returns:
        Labelled ``=== stdout ===`` / ``=== stderr ===`` / ``=== exit code ===``
        sections.
    """
    python = venv_python_path or sys.executable
    result = subprocess.run(  # noqa: S603
        [python, file_path],
        capture_output=True, text=True, check=False, env=_merged_env(env),
    )
    return (
        f"=== stdout ===\n{result.stdout}\n"
        f"=== stderr ===\n{result.stderr}\n"
        f"=== exit code ===\n{result.returncode}"
    )


def execute_python_snippet(
    code: str,
    venv_python_path: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Run an arbitrary Python snippet and return stdout, stderr and exit code.

    Args:
        code: Python source to execute (may be multi-line).
        venv_python_path: Python executable to use.
        env: Extra environment variables merged over the current environment.

    Returns:
        Same labelled format as :func:`execute_python_file`.
    """
    python = venv_python_path or sys.executable
    result = subprocess.run(  # noqa: S603
        [python, "-c", code],
        capture_output=True, text=True, check=False, env=_merged_env(env),
    )
    return (
        f"=== stdout ===\n{result.stdout}\n"
        f"=== stderr ===\n{result.stderr}\n"
        f"=== exit code ===\n{result.returncode}"
    )


def execute_notebook(
    notebook_path: str,
    venv_python_path: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    """Execute every cell of a Jupyter notebook and return code + outputs.

    Cells that raise are reported but execution continues. Rich objects are
    rendered to text; images become ``[image output]`` placeholders.

    Args:
        notebook_path: Absolute path to the ``.ipynb`` file.
        venv_python_path: Python executable whose kernel should run the cells.
        env: Extra environment variables for the kernel process.

    Returns:
        Per-cell code/output blocks, or ``"[error] ..."`` on a fatal problem.
    """
    try:
        import nbformat
        from nbclient import NotebookClient
    except Exception as exc:  # noqa: BLE001
        return f"[error] notebook execution requires nbclient/nbformat: {exc}"

    try:
        nb = nbformat.read(notebook_path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(Path(notebook_path).parent)}},
        )
        if env:
            os.environ.update(env)
        client.execute()
    except Exception as exc:  # noqa: BLE001
        return f"[error] notebook execution failed: {exc}"

    blocks: list[str] = []
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        out_text = _render_cell_outputs(cell.get("outputs", []))
        blocks.append(
            f"=== Cell {i} ===\n--- code ---\n{cell.source}\n--- output ---\n{out_text}"
        )
    return "\n\n".join(blocks) if blocks else "[no code cells]"


def _render_cell_outputs(outputs: list[dict]) -> str:
    """Flatten a notebook cell's outputs into plain text."""
    parts: list[str] = []
    for out in outputs:
        otype = out.get("output_type")
        if otype == "stream":
            parts.append(out.get("text", ""))
        elif otype in {"execute_result", "display_data"}:
            data = out.get("data", {})
            if "text/plain" in data:
                parts.append(data["text/plain"])
            if "image/png" in data or "image/jpeg" in data:
                parts.append("[image output]")
        elif otype == "error":
            parts.append("\n".join(out.get("traceback", [])))
    return "".join(parts).strip() or "[no output]"
