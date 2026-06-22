"""doc_agent: an agentic documentation-QA system with calibrated confidence.

Public entry points:
    >>> from doc_agent import DocAgent, LocalExecutionBackend, QuestionSet, QuestionItem
    >>> agent = DocAgent(backend=LocalExecutionBackend(workdir="./work"))
    >>> result = agent.answer_questions(QuestionSet(
    ...     repo_url="https://gitlab.com/example/project.git",
    ...     questions=[QuestionItem(id="q1", question="What does train() return?",
    ...                             requires_code_execution=True)],
    ... ))
    >>> for a in result.answers:
    ...     print(a.question_id, a.confidence, a.answer)
"""

from .agent import DocAgent
from .config import Settings
from .confidence import ConfidenceCalibrator, PlattScaler
from .models import (
    AgentAnswer,
    AgentResult,
    ConfidenceLabel,
    ConfidenceSignals,
    Evidence,
    ExecutionResult,
    QuestionItem,
    QuestionSet,
    SourceType,
)
from .tools import ExecutionBackend, LocalExecutionBackend

__all__ = [
    "DocAgent",
    "Settings",
    "ConfidenceCalibrator",
    "PlattScaler",
    "AgentAnswer",
    "AgentResult",
    "ConfidenceLabel",
    "ConfidenceSignals",
    "Evidence",
    "ExecutionResult",
    "QuestionItem",
    "QuestionSet",
    "SourceType",
    "ExecutionBackend",
    "LocalExecutionBackend",
]

__version__ = "0.1.0"
