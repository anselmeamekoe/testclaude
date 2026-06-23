"""Documentation-QA agent package.

Public API:

* :class:`~doc_qa_agent.agent.DocQAAgent` — the orchestrator.
* :class:`~doc_qa_agent.models.QuestionSet` / :class:`~doc_qa_agent.models.AgentResult`
  — the imposed input/output schemas.
* :func:`~doc_qa_agent.config.get_settings` — configuration loader.
"""

from __future__ import annotations

from .agent import DocQAAgent
from .config import Settings, get_settings
from .models import (
    AgentResult,
    AnswerConfidence,
    Evidence,
    QuestionAnswer,
    QuestionSet,
)

__all__ = [
    "DocQAAgent",
    "Settings",
    "get_settings",
    "QuestionSet",
    "AgentResult",
    "QuestionAnswer",
    "AnswerConfidence",
    "Evidence",
]
