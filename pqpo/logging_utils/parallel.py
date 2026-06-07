"""Parallel map over API-bound work with progress.

Target-model calls are I/O-bound, so threads give real speedups on real providers.
For offline/simulated workloads, a process backend is also available for true
multiprocessing. Results are returned in input order. With workers<=1 this is a
plain sequential map (identical results), so determinism is preserved when you
need it.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

from .progress import titer


def parallel_map(fn: Callable, items: Iterable, workers: int = 1,
                 desc: str = "", total: int = None, progress: bool = True,
                 backend: str = "thread") -> list:
    """Map fn over items with thread or process workers, preserving order.

    progress=False suppresses the tqdm bar (use when another live view — e.g. the
    run_mvp rich dashboard — already owns the terminal)."""
    items = list(items)
    total = total if total is not None else len(items)
    wrap = (lambda it: titer(it, desc=desc, total=total)) if progress else (lambda it: it)
    if workers <= 1:
        return [fn(x) for x in wrap(items)]
    if backend not in {"thread", "process"}:
        raise ValueError("parallel backend must be 'thread' or 'process'")
    results = [None] * len(items)
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with executor_cls(max_workers=workers) as pool:
        fut_to_i = {pool.submit(fn, x): i for i, x in enumerate(items)}
        for fut in wrap(as_completed(fut_to_i)):
            results[fut_to_i[fut]] = fut.result()
    return results


def thread_map(fn: Callable, items: Iterable, workers: int = 1,
               desc: str = "", total: int = None, progress: bool = True) -> list:
    """Backward-compatible thread-only wrapper."""
    return parallel_map(fn, items, workers=workers, desc=desc, total=total,
                        progress=progress, backend="thread")
