"""Injected dependencies, as `Annotated` aliases.

FastAPI's own `Depends` is the DI container. A third-party one would add a second object lifecycle
to reason about for exactly two providers, and neither of them needs more than request scope.

The aliases are the documented FastAPI style: the dependency lives in the *type*, not in a mutable
default argument, so a signature can be reused across routes and Ruff's B008
(function-call-in-default-argument) has nothing to fire on. `app.dependency_overrides[get_db]` still
works — the override key is the provider callable, which is unchanged.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from argotech.data.db import get_db
from argotech.models.registry import ModelManager
from argotech.serving.container import model_manager


def get_model_manager() -> ModelManager:
    return model_manager


ModelManagerDep = Annotated[ModelManager, Depends(get_model_manager)]
DbSessionDep = Annotated[Session, Depends(get_db)]
