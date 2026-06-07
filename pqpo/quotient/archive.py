"""PhenotypeArchive: online cell assignment for closed-loop PQPO (Sec 2.6).

Maintains a growing set of phenotype cells. New prompts are assigned to the
nearest existing cell if within tau of its medoid fingerprint, else they seed a
new (previously unoccupied) cell. Tracks labelled scores per cell and supports
periodic reclustering and stability logging.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from ..data.datastructures import BehaviorFingerprint, CellState
from ..fingerprints.distances import fingerprint_distance


class PhenotypeArchive:
    def __init__(self, tau: float, weights: dict = None):
        self.tau = tau
        self.weights = weights
        self.fingerprints: dict[str, BehaviorFingerprint] = {}
        self.prompt_text: dict[str, str] = {}
        self.cell_of: dict[str, str] = {}              # prompt_id -> cell_id
        self.cell_members: dict[str, list[str]] = {}   # cell_id -> [prompt_id]
        self.cell_medoid_fp: dict[str, BehaviorFingerprint] = {}
        self.cell_state: dict[str, CellState] = {}
        self.iteration = 0
        self._next_cell = 0
        self.new_cell_events = 0
        self.assign_events = 0
        self.target_hits = 0
        self.target_attempts = 0

    # -- assignment --------------------------------------------------------- #
    def assign_or_create_cell(self, prompt_id: str, fp: BehaviorFingerprint,
                              prompt_text: str = "", intended_cell: str = None) -> str:
        self.fingerprints[prompt_id] = fp
        self.prompt_text[prompt_id] = prompt_text
        best_cell, best_d = None, np.inf
        for cid, mfp in self.cell_medoid_fp.items():
            d = fingerprint_distance(fp, mfp, self.weights)
            if d < best_d:
                best_cell, best_d = cid, d
        if best_cell is not None and best_d <= self.tau:
            cid = best_cell
            self.cell_members[cid].append(prompt_id)
            self.assign_events += 1
            new_cell = False
        else:
            cid = f"cell_{self._next_cell:04d}"
            self._next_cell += 1
            self.cell_members[cid] = [prompt_id]
            self.cell_medoid_fp[cid] = fp
            self.cell_state[cid] = CellState(cid, [prompt_id], prompt_id)
            self.new_cell_events += 1
            new_cell = True
        self.cell_of[prompt_id] = cid
        self.cell_state.setdefault(cid, CellState(cid, [], prompt_id))
        if prompt_id not in self.cell_state[cid].prompt_ids:
            self.cell_state[cid].prompt_ids.append(prompt_id)
        # target-cell hit accounting
        if intended_cell is not None:
            self.target_attempts += 1
            if (not new_cell and cid == intended_cell) or \
               (intended_cell == "__new__" and new_cell):
                self.target_hits += 1
        return cid

    # -- scoring ------------------------------------------------------------ #
    def update_scores(self, cell_id: str, prompt_id: str, scores: list[float]) -> None:
        st = self.cell_state[cell_id]
        for s in scores:
            st.record(prompt_id, s)

    def occupancy_distribution(self) -> dict[str, float]:
        total = sum(len(m) for m in self.cell_members.values()) or 1
        return {cid: len(m) / total for cid, m in self.cell_members.items()}

    def best_prompt(self) -> Optional[str]:
        best_pid, best_score = None, -np.inf
        for cid, st in self.cell_state.items():
            if st.n_labeled_evals == 0:
                continue
            for pid, scs in st.per_prompt_scores.items():
                m = float(np.mean(scs))
                if m > best_score:
                    best_pid, best_score = pid, m
        if best_pid is None and self.cell_of:
            return next(iter(self.cell_of))
        return best_pid

    @property
    def new_cell_entry_rate(self) -> float:
        total = self.new_cell_events + self.assign_events
        return self.new_cell_events / total if total else 0.0

    @property
    def target_cell_hit_rate(self) -> float:
        return self.target_hits / self.target_attempts if self.target_attempts else 0.0

    def recluster_if_needed(self):
        # Recompute each cell's medoid fingerprint from current members.
        for cid, members in self.cell_members.items():
            if len(members) <= 1:
                continue
            fps = [self.fingerprints[m] for m in members]
            best, best_d = members[0], np.inf
            for i, mi in enumerate(members):
                d = float(np.mean([fingerprint_distance(fps[i], f, self.weights) for f in fps]))
                if d < best_d:
                    best, best_d = mi, d
            self.cell_medoid_fp[cid] = self.fingerprints[best]
