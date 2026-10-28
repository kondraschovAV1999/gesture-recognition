import os
import shutil

from hand_gesture_recognition.utils.paths import (
    PROJECT_ROOT,
    resolve_path,
    resolve_tracking_uri,
)


def test_resolve_path_anchors_relative_to_project_root():
    assert resolve_path("output") == os.path.join(PROJECT_ROOT, "output")
    # absolute paths pass through unchanged
    assert resolve_path("C:/some/abs/path") == "C:/some/abs/path"


def test_resolve_tracking_uri_anchors_relative_sqlite():
    uri = resolve_tracking_uri("sqlite:///mlflow.db")
    assert uri.startswith("sqlite:///")
    # The database path is now anchored to the project root (not CWD).
    assert uri != "sqlite:///mlflow.db"
    assert PROJECT_ROOT.replace(os.sep, "/") in uri


def test_resolve_tracking_uri_keeps_memory_and_absolute_sqlite():
    assert resolve_tracking_uri("sqlite:///:memory:") == "sqlite:///:memory:"
    abs_uri = f"sqlite:///{PROJECT_ROOT.replace(os.sep, '/')}/mlflow.db"
    assert resolve_tracking_uri(abs_uri) == abs_uri


def test_resolve_tracking_uri_file_store_and_remote():
    # scheme-less relative path -> anchored
    assert resolve_tracking_uri("mlruns") == os.path.join(PROJECT_ROOT, "mlruns")
    # scheme-less absolute path -> unchanged
    assert resolve_tracking_uri("C:/mlruns") == "C:/mlruns"
    # remote URIs -> unchanged
    assert resolve_tracking_uri("http://localhost:5000") == "http://localhost:5000"
    assert (
        resolve_tracking_uri("postgresql://user:pass@host/db")
        == "postgresql://user:pass@host/db"
    )
