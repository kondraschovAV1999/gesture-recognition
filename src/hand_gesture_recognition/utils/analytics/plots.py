from __future__ import annotations

import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from hand_gesture_recognition.utils.datasets import Dataset


def plot_class_distribution(dataset: Dataset, path: str) -> None:
    """Save a bar chart of class counts to `path`."""
    counts = Counter(int(label) for label in dataset.labels)
    names = [dataset.metadata.class_names[k] for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]
    fig, ax = plt.subplots()
    ax.bar(names, values)
    ax.set_title(f"Class distribution — {dataset.metadata.name}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
