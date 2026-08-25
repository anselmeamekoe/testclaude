"""
Shared data contract for the toolkit.

The whole toolkit talks to ONE object: `Dataset`. Everything else (complexity,
learning curves, weak points) consumes a `Dataset`. This keeps the analyzers
model- and loader-agnostic.

Your data arrives as a "pydantic dataloader". We do not assume its exact shape:
any object that can hand us a pandas DataFrame is accepted (a DataFrame, a
callable returning one, or an object exposing `.load()/.to_frame()/.dataframe`).
See `Dataset.from_loaders`.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------------- #
# Model & loader protocols                                                     #
# --------------------------------------------------------------------------- #
@runtime_checkable
class Estimator(Protocol):
    """Anything scikit-learn-shaped."""

    def fit(self, X: Any, y: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...


LoaderLike = Union[pd.DataFrame, Callable[[], pd.DataFrame], Any]


def _as_frame(loader: LoaderLike) -> pd.DataFrame:
    """Coerce a 'pydantic dataloader' (or anything reasonable) into a DataFrame."""
    if isinstance(loader, pd.DataFrame):
        return loader.copy()
    if callable(loader):
        return _as_frame(loader())
    for attr in ("load", "to_frame", "dataframe", "df", "data"):
        if hasattr(loader, attr):
            obj = getattr(loader, attr)
            obj = obj() if callable(obj) else obj
            if isinstance(obj, pd.DataFrame):
                return obj.copy()
    # pydantic v2 BaseModel list -> DataFrame
    if hasattr(loader, "model_dump"):
        return pd.json_normalize(loader.model_dump())
    if isinstance(loader, Sequence) and len(loader) and hasattr(loader[0], "model_dump"):
        return pd.DataFrame([r.model_dump() for r in loader])
    raise TypeError(
        f"Could not turn {type(loader)!r} into a DataFrame. Pass a DataFrame, a "
        "callable returning one, or an object with .load()/.to_frame()/.dataframe."
    )


class Task(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


# --------------------------------------------------------------------------- #
# Config (pydantic) — the 'no-code' knobs live here                            #
# --------------------------------------------------------------------------- #
class EvalConfig(BaseModel):
    """All tunables in one validated place; every analyzer accepts one."""

    model_config = ConfigDict(extra="forbid")

    task: Task = Task.CLASSIFICATION
    random_state: int = 42
    n_repeats: int = 12            # Monte-Carlo repeats (edf, stability, rademacher)
    n_bootstrap: int = 20          # bootstrap resamples for prediction variance
    max_rows_for_refit: int = 4000  # subsample cap to keep refit-heavy stats cheap
    cv_folds: int = 4
    # weak-point discovery
    max_slice_depth: int = 3       # depth of the interpretable "error tree"
    min_slice_frac: float = 0.03   # ignore slices smaller than this share of test
    # temporal / drift
    time_col: Optional[str] = None
    n_windows: int = 8

    @field_validator("min_slice_frac")
    @classmethod
    def _frac(cls, v):
        if not 0 < v < 1:
            raise ValueError("min_slice_frac must be in (0, 1)")
        return v


# --------------------------------------------------------------------------- #
# Dataset — the single object the toolkit consumes                             #
# --------------------------------------------------------------------------- #
class Dataset(BaseModel):
    """Immutable-ish container. Holds numeric feature matrices + targets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]
    task: Task = Task.CLASSIFICATION
    time_train: Optional[pd.Series] = None
    time_test: Optional[pd.Series] = None

    # ---- constructors ---------------------------------------------------- #
    @classmethod
    def from_frames(
        cls,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str,
        task: Union[Task, str] = Task.CLASSIFICATION,
        time_col: Optional[str] = None,
        features: Optional[Sequence[str]] = None,
    ) -> "Dataset":
        task = Task(task)
        drop = {target} | ({time_col} if time_col else set())
        feats = list(features) if features else [c for c in train.columns if c not in drop]
        return cls(
            X_train=train[feats].reset_index(drop=True),
            y_train=train[target].reset_index(drop=True),
            X_test=test[feats].reset_index(drop=True),
            y_test=test[target].reset_index(drop=True),
            feature_names=feats,
            task=task,
            time_train=(train[time_col].reset_index(drop=True) if time_col else None),
            time_test=(test[time_col].reset_index(drop=True) if time_col else None),
        )

    @classmethod
    def from_loaders(
        cls,
        train_loader: LoaderLike,
        test_loader: LoaderLike,
        target: str,
        task: Union[Task, str] = Task.CLASSIFICATION,
        time_col: Optional[str] = None,
        features: Optional[Sequence[str]] = None,
    ) -> "Dataset":
        """Entry point for a pydantic dataloader (or any DataFrame-yielding thing)."""
        return cls.from_frames(
            _as_frame(train_loader), _as_frame(test_loader),
            target=target, task=task, time_col=time_col, features=features,
        )

    # ---- convenience ----------------------------------------------------- #
    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    def numpy(self):
        return (
            self.X_train.to_numpy(dtype=float),
            self.y_train.to_numpy(),
            self.X_test.to_numpy(dtype=float),
            self.y_test.to_numpy(),
        )

    def subsample_train(self, n: int, rng: np.random.Generator):
        """Return (Xs, ys) subsample of train for cheap refit-heavy stats."""
        if self.n_train <= n:
            return self.X_train.to_numpy(float), self.y_train.to_numpy()
        idx = rng.choice(self.n_train, size=n, replace=False)
        return self.X_train.to_numpy(float)[idx], self.y_train.to_numpy()[idx]
