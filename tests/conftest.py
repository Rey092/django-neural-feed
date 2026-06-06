import pytest
import threading
import numpy as np
from django.conf import settings
from django.db import transaction, connection
from django_neural_feed.signals import register_feed_signals, register_content_signals
from tests.feeds import TestParentFeed
from tests.models import TestArticle


class MockTestingEncoder:
    @classmethod
    def text_to_vector(cls, text: str, model_name: str) -> list[float]:
        return [0.5, 0.5, 0.5]

    @classmethod
    def average_vectors(cls, vectors: list[list[float]], limit: int) -> list[float]:
        if not vectors:
            return []
        arr = np.asarray(vectors[:limit], dtype=np.float32)
        if arr.size == 0:
            return []
        mean = np.mean(arr, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        return mean.tolist()


class SyncTestThread:
    def __init__(self, target, args=(), kwargs=None, **dummy_kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self):
        self.target(*self.args, **self.kwargs)


@pytest.fixture(autouse=True)
def setup_test_environment(db, monkeypatch):
    """Prepares mock settings, intercepts async execution, and protects connection."""

    dnf_settings = getattr(settings, "DJANGO_NEURAL_FEED", {})
    dnf_settings["ENCODER_CLASS"] = MockTestingEncoder
    dnf_settings["VECTOR_DIMENSION"] = 3

    monkeypatch.setattr(settings, "DJANGO_NEURAL_FEED", dnf_settings)

    monkeypatch.setattr(transaction, "on_commit", lambda func: func())

    monkeypatch.setattr(threading, "Thread", SyncTestThread)

    monkeypatch.setattr(connection, "close", lambda: None)

    register_feed_signals(TestParentFeed)
    register_content_signals(TestArticle)
