"""
Agent step log: one JSON object per line (JSON Lines).

Each playlist request gets a run id, so every step of one request can be
pulled out for ai_interactions.md and the model card.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_LOG_PATH = "logs/agent_steps.jsonl"


class AgentLogger:
    def __init__(self, log_path: Optional[str] = DEFAULT_LOG_PATH):
        """log_path=None keeps steps in memory only (used by tests)."""
        self.log_path = log_path
        self.run_id: Optional[str] = None
        self.steps: List[Dict[str, Any]] = []

    def start_run(self) -> str:
        self.run_id = uuid.uuid4().hex[:12]
        self.steps = []
        return self.run_id

    def log_step(self, step: str, payload: Dict[str, Any]) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "step": step,
            **payload,
        }
        self.steps.append(record)
        if self.log_path:
            path = Path(self.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
