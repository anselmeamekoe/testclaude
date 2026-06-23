"""Repository acquisition — the imposed GitLab clone helper.

This implements the exact signature the organizers specified for cloning a
private GitLab repository over HTTPS using a token embedded in the URL.
"""

from __future__ import annotations

import os
import subprocess
from urllib.parse import urlparse, urlunparse


def clone(gitlab_token: str, repo_url: str, dest_folder: str) -> None:
    """Clone a private GitLab repository over HTTPS using token authentication.

    The token is injected into the URL's net-location (``user:<token>@host``)
    so ``git`` can authenticate non-interactively.

    Args:
        gitlab_token: A GitLab access token with read permission on the repo.
        repo_url: HTTPS URL of the repository (``ssh://`` and ``git://`` are
            rejected because token-in-URL auth only applies to HTTPS).
        dest_folder: Destination directory; created if it does not exist.

    Raises:
        ValueError: If ``repo_url`` is not an HTTPS URL.
        RuntimeError: If ``git clone`` exits with a non-zero status.
    """
    parsed = urlparse(repo_url)
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS GitLab URLs are supported for token authentication")

    netloc = f"user:{gitlab_token}@{parsed.netloc}"
    auth_url = urlunparse(parsed._replace(netloc=netloc))

    os.makedirs(dest_folder, exist_ok=True)
    cmd = ["git", "clone", auth_url, dest_folder]
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Git clone failed (exit code {result.returncode}).\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
