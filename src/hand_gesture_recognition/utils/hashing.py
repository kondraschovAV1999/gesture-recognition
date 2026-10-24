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
            if name == ".content_hash":
                continue  # the cache sidecar is not part of the dataset bytes
            abs_path = os.path.join(dirpath, name)
            entries.append((os.path.relpath(abs_path, root), abs_path))
    for rel_path, abs_path in sorted(entries):
        _update_hash_with_file(h, rel_path, abs_path)
    return h.hexdigest()


def hash_directory_cached(root: str) -> str:
    """Hash every file under root, caching the result in `<root>/.content_hash`.

    The first call computes the SHA-256 (potentially slow for large datasets)
    and writes it to the sidecar; subsequent calls read the cached value.
    Delete the `.content_hash` sidecar to force a recomputation (e.g. after
    changing the dataset files). The sidecar is excluded from the hash itself.
    """
    cache = os.path.join(root, ".content_hash")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return fh.read().strip()
    digest = hash_directory(root)
    with open(cache, "w", encoding="utf-8") as fh:
        fh.write(digest + "\n")
    return digest
