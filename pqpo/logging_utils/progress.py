"""Real-time terminal progress + live metrics (rich + tqdm).

Design
  * `run_mvp.py` uses the rich path (a Live view = progress bars + a metrics table
    that updates as each selector result lands).
  * `build_candidate_pool.py` / `run_closed_loop.py` use tqdm bars + rich tables.
  Keeping rich.Live and tqdm in separate scripts avoids terminal contention.

Everything degrades gracefully: if a library is missing or stdout is not a TTY,
calls become no-ops / plain prints, so headless/CI runs still work.
"""
from __future__ import annotations

import sys
from typing import Iterable, Optional

try:
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                               SpinnerColumn, TextColumn, TimeElapsedColumn)
    from rich.table import Table
    from rich.text import Text
    _HAS_RICH = True
    console = Console()
except Exception:  # pragma: no cover
    _HAS_RICH = False
    console = None


# --------------------------------------------------------------------------- #
# tqdm helper (for scripts without an active rich.Live)
# --------------------------------------------------------------------------- #
def titer(iterable: Iterable, desc: str = "", total: Optional[int] = None,
          leave: bool = False):
    if _HAS_TQDM:
        return _tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True)
    return iterable


class _NullBar:
    def update(self, n=1): pass
    def set_postfix(self, *a, **k): pass
    def close(self): pass


def make_bar(total: int, desc: str = ""):
    """A standalone tqdm bar (for callback-driven loops). No-op if tqdm absent."""
    if _HAS_TQDM:
        return _tqdm(total=total, desc=desc, leave=False, dynamic_ncols=True)
    return _NullBar()


def log(msg: str):
    if _HAS_RICH:
        console.print(msg)
    else:
        print(msg)


def rule(title: str):
    if _HAS_RICH:
        console.rule(f"[bold cyan]{title}")
    else:
        print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


# --------------------------------------------------------------------------- #
# Live dashboard for run_mvp: progress bars + a per-method metrics table
# --------------------------------------------------------------------------- #
class MVPDashboard:
    """Live view: an overall bar, a per-task bar, and a table of running mean
    held-out score per method x budget that refreshes as results arrive."""

    def __init__(self, methods: list[str], budgets: list[int], tasks: list[str],
                 total_steps: int):
        self.methods = methods
        self.budgets = budgets
        self.tasks = tasks
        self.enabled = _HAS_RICH and sys.stdout.isatty()
        # state[method][budget] = list of held-out means across tasks/seeds
        self.state = {m: {b: [] for b in budgets} for m in methods}
        self.current_task = ""
        self.cond_lines: list[str] = []
        if not self.enabled:
            self._fallback_total = total_steps
            self._fallback_done = 0
            return
        self.progress = Progress(
            SpinnerColumn(), TextColumn("[bold]{task.description}"),
            BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
            console=console, transient=False)
        self.overall = self.progress.add_task("overall", total=total_steps)
        self.fp_task = self.progress.add_task("fingerprinting", total=1, visible=False)
        self.live = Live(self._render(), console=console, refresh_per_second=8)

    # -- lifecycle ---------------------------------------------------------- #
    def __enter__(self):
        if self.enabled:
            self.live.__enter__()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self.live.update(self._render())
            self.live.__exit__(*exc)

    # -- updates ------------------------------------------------------------ #
    def start_task(self, task_id: str):
        self.current_task = task_id
        if self.enabled:
            self.live.update(self._render())

    def fingerprint_bar(self, total: int):
        if self.enabled:
            self.progress.reset(self.fp_task, total=total, visible=True)
            self.progress.update(self.fp_task,
                                 description=f"fingerprint {self.current_task}")

    def fingerprint_advance(self):
        if self.enabled:
            self.progress.advance(self.fp_task)

    def fingerprint_done(self):
        if self.enabled:
            self.progress.update(self.fp_task, visible=False)

    def record(self, method: str, budget: int, heldout: float):
        if method in self.state and budget in self.state[method]:
            self.state[method][budget].append(heldout)

    def step(self):
        if self.enabled:
            self.progress.advance(self.overall)
            self.live.update(self._render())
        else:
            self._fallback_done += 1
            if self._fallback_done % 5 == 0 or self._fallback_done == self._fallback_total:
                print(f"  ... {self._fallback_done}/{self._fallback_total} steps")

    def set_conditions(self, lines: list[str]):
        self.cond_lines = lines
        if self.enabled:
            self.live.update(self._render())

    # -- rendering ---------------------------------------------------------- #
    def _mean(self, method, budget):
        vals = self.state[method][budget]
        return sum(vals) / len(vals) if vals else None

    def _render(self):
        if not self.enabled:
            return None
        table = Table(title=f"Running held-out score  (task: {self.current_task or '—'})",
                      expand=False)
        table.add_column("method", style="bold")
        for b in self.budgets:
            table.add_column(f"B={b}", justify="right")
        # rank methods by their best-budget running mean, PQPO highlighted
        last_b = self.budgets[-1]
        order = sorted(self.methods,
                       key=lambda m: (self._mean(m, last_b) or -1), reverse=True)
        for m in order:
            row = [Text(m, style="bold magenta") if m == "pqpo" else m]
            for b in self.budgets:
                v = self._mean(m, b)
                cell = "—" if v is None else f"{v:.3f}"
                style = "green" if (m == "pqpo" and v is not None) else ""
                row.append(Text(cell, style=style))
            table.add_row(*row)
        renderables = [self.progress, table]
        if self.cond_lines:
            renderables.append(Panel("\n".join(self.cond_lines),
                                     title="breakthrough conditions",
                                     border_style="cyan"))
        return Group(*renderables)


def metrics_table(title: str, columns: list[str], rows: list[list[str]]):
    """One-shot rich table (used by the tqdm scripts at the end)."""
    if not _HAS_RICH:
        print(f"\n{title}")
        print(" | ".join(columns))
        for r in rows:
            print(" | ".join(str(x) for x in r))
        return
    t = Table(title=title)
    for c in columns:
        t.add_column(c)
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console.print(t)
