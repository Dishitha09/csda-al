"""JSONL events with deterministic sequence numbers and finite payloads."""
from pathlib import Path
from typing import Any, Protocol
import json


class Logger(Protocol):
    def log(self, event: str, **payload: Any) -> None: ...


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sequence = len(self.path.read_text(encoding="utf-8").splitlines()) if self.path.exists() else 0

    def log(self, event: str, **payload: Any) -> None:
        record = {"sequence": self.sequence, "event": event, **payload}
        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        self.sequence += 1
