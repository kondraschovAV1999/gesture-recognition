from __future__ import annotations

import hashlib
import os
from typing import Any, Iterable, List


def _update_hash_with_file(h: Any, path_label: str, abs_path: str) -> None:
    h.update(path_label.encode("utf-8", errors="replace"))
    with open(abs_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)


def hash_files(paths: Iterable[str]) -> str:
    """SHA-256 over each path string plus its bytes (paths hashed as given)."""
    h = hashlib.sha256()
    for p in sorted(paths):
        _update_hash_with_file(h, p, p)
    return h.hexdigest()


def hash_directory(root: str) -> str:
    """Location-independent SHA-256 over (relative path, bytes) for every file under root.

    Paths are recorded relative to root, so identical bytes produce the same
    hash regardless of where the directory is stored.
    """
    h = hashlib.sha256()
    entries: List[tuple] = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            abs_path = os.path.join(dirpath, name)
            entries.append((os.path.relpath(abs_path, root), abs_path))
    for rel_path, abs_path in sorted(entries):
        _update_hash_with_file(h, rel_path, abs_path)
    return h.hexdigest()
