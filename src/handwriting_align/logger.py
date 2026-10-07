"""Persist backend numerical metrics without an external tracking service."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from flow_factory.logger.abc import Logger


class MetricJSONLogger(Logger):
    def _init_platform(self) -> None:
        self.platform = None
        self.path = Path(self.config.log_args.save_dir) / str(self.config.log_args.run_name) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def _convert_to_platform(self, value, height=None, width=None):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool, dict)):
            return value
        return {"artifact_type": type(value).__name__}

    def _log_impl(self, data, step: int) -> None:
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "step": step, "metrics": data}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print("METRIC_JSON " + line, flush=True)
