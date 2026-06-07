"""Automatic cost accounting (Sec 3.4).

The ledger ingests every APITrace and aggregates by run and by method. It keeps
the protocol-critical separation between call types so that the paper can report
sentinel cost explicitly (never report "labeled evals saved" without sentinel
cost, per Sec 5.4).
"""
from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from typing import Optional

from ..data.datastructures import APITrace

CALL_TYPES = ["proposal", "reflection", "sentinel", "selector_dev", "final_test"]
# Optimization cost = everything except final_test (which is evaluation).
OPTIMIZATION_CALL_TYPES = ["proposal", "reflection", "sentinel", "selector_dev"]


class CostLedger:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.traces: list[APITrace] = []
        self._lock = threading.Lock()       # safe under parallel workers
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def record_api_call(self, trace: APITrace) -> None:
        with self._lock:
            self.traces.append(trace)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def _agg(self, traces: list[APITrace]) -> dict:
        agg = {f"{ct}_calls": 0 for ct in CALL_TYPES}
        agg.update(
            input_tokens=0, output_tokens=0, total_tokens=0,
            dollar_cost=0.0, latency_ms=0.0, cache_hits=0, cache_misses=0,
            n_calls=0,
        )
        for t in traces:
            agg["n_calls"] += 1
            key = f"{t.call_type}_calls"
            if key in agg:
                agg[key] += 1
            agg["input_tokens"] += t.input_tokens
            agg["output_tokens"] += t.output_tokens
            agg["total_tokens"] += t.total_tokens
            agg["dollar_cost"] += t.dollar_cost
            agg["latency_ms"] += t.latency_ms
            agg["cache_hits"] += int(t.cache_hit)
            agg["cache_misses"] += int(not t.cache_hit)
        agg["optimization_dollar_cost"] = sum(
            t.dollar_cost for t in traces if t.call_type in OPTIMIZATION_CALL_TYPES
        )
        agg["final_test_dollar_cost"] = sum(
            t.dollar_cost for t in traces if t.call_type == "final_test"
        )
        return agg

    def aggregate_by_run(self, run_id: str) -> dict:
        return self._agg([t for t in self.traces if t.run_id == run_id])

    def aggregate_by_method(self, method_name: str) -> dict:
        return self._agg([t for t in self.traces if t.run_id.startswith(method_name)])

    def aggregate_all(self) -> dict:
        return self._agg(self.traces)

    def export_cost_table(self, path: str) -> None:
        by_run: dict[str, list[APITrace]] = defaultdict(list)
        for t in self.traces:
            by_run[t.run_id].append(t)
        rows = [{"run_id": rid, **self._agg(ts)} for rid, ts in by_run.items()]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
