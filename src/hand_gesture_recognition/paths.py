from __future__ import annotations

import os

# Absolute path to the project (repo) root, derived from this file's location
# under src/hand_gesture_recognition/. Robust regardless of the current
# working directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(path: str) -> str:
    """Return `path` as an absolute path, resolving relative paths against PROJECT_ROOT."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)
