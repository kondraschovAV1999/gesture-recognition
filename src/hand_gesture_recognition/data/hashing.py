from __future__ import annotations

import hashlib
import os
from typing import Iterable, List


def hash_files(paths: Iterable[str]) -> str:
    """SHA-256 over the concatenation of (relative path, bytes) pairs."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode("utf-8", errors="replace"))
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def hash_directory(root: str) -> str:
    """Deterministic hash of every file under root, sorted by relative path."""
    rel_files: List[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            rel_files.append(os.path.join(dirpath, name))
    return hash_files(rel_files)
