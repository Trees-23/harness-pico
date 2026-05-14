"""Fine-grained locks for serialized file writes."""

from __future__ import annotations

import threading
from collections.abc import Hashable
from contextlib import contextmanager
from typing import Iterator


class FileLockManager:
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[Hashable, threading.Lock] = {}

    @contextmanager
    def lock(self, key: Hashable) -> Iterator[None]:
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            yield

