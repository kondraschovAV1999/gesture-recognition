from __future__ import annotations

import os

# Absolute path to the project (repo) root, derived from this file's location
# under src/hand_gesture_recognition/utils/. Robust regardless of the current
# working directory.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def resolve_path(path: str) -> str:
    """Return `path` as an absolute path, resolving relative paths against PROJECT_ROOT."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def resolve_tracking_uri(uri: str) -> str:
    """Anchor a relative MLflow tracking URI to :data:`PROJECT_ROOT`.

    Handles the two local-store forms MLflow supports:

    - A scheme-less URI (a local file store, e.g. ``mlruns``) is treated as a
      relative filesystem path and anchored to ``PROJECT_ROOT``.
    - A ``sqlite:///...`` URI with a relative database path is anchored too
      (``sqlite:///mlflow.db`` -> ``sqlite:///<PROJECT_ROOT>/mlflow.db``).

    Absolute paths and remote URIs (``http://``, ``postgresql://``, ...) pass
    through unchanged, as does an in-memory sqlite database (``:memory:``).
    """
    if "://" not in uri:
        # Local file-store path.
        return uri if os.path.isabs(uri) else os.path.join(PROJECT_ROOT, uri)
    if uri.startswith("sqlite:///"):
        db_path = uri[len("sqlite:///") :]
        if db_path and db_path != ":memory:" and not os.path.isabs(db_path):
            abs_path = os.path.join(PROJECT_ROOT, db_path).replace(os.sep, "/")
            return f"sqlite:///{abs_path}"
    return uri
