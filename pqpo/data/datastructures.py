"""Core data structures for PQPO (Phenotype Quotient Prompt Optimization).

These dataclasses are the on-disk and in-memory contracts used across the whole
pipeline. They follow Section 3.2 of the implementation plan. Every structure is
JSON-serialisable via :func:`to_dict` so that artifacts can be replayed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Any


# --------------------------------------------------------------------------- #
# Task-level objects
# --------------------------------------------------------------------------- #
@dataclass
class TaskExample:
    """A single labelled (or unlabelled) example for a task."""
    example_id: str
    task_id: str
    input_text: str
    target: Optional[str] = None          # gold label / answer; None for sentinels
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CandidatePrompt:
    """A candidate system/instruction prompt produced by some generator."""
    prompt_id: str
    task_id: str
    source_method: str                    # GEPA, MIPROv2, CAPO, APE, paraphrase, ...
    prompt_text: str
    parent_prompt_id: Optional[str] = None
    proposal_call_ids: list[str] = field(default_factory=list)
    reflection_call_ids: list[str] = field(default_factory=list)
    token_length: int = 0
    generation_seed: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# API / execution objects
# --------------------------------------------------------------------------- #
@dataclass
class APITrace:
    """A single target-model call. This is the unit of cost accounting."""
    call_id: str
    run_id: str
    model_provider: str
    model_name: str
    task_id: str
    call_type: str                        # proposal|reflection|sentinel|selector_dev|final_test
    input_text_hash: str
    full_input_text: str
    output_text: str
    temperature: float
    max_output_tokens: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    dollar_cost: float
    cache_hit: bool
    request_hash: str
    model_version: Optional[str] = None
    timestamp_utc: Optional[str] = None
    prompt_id: Optional[str] = None
    example_id: Optional[str] = None
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedOutput:
    """Result of running an OutputNormalizer over a raw model output."""
    normalized_answer: Optional[str]
    format_valid: bool
    refusal_flag: bool
    error_type: Optional[str]
    tool_sequence: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SentinelOutput:
    """A target-model output on a single unlabelled sentinel probe."""
    prompt_id: str
    task_id: str
    sentinel_id: str
    call_id: str
    raw_output: str
    normalized_answer: Optional[str]
    format_valid: bool
    refusal_flag: bool
    parse_error_type: Optional[str]
    output_tokens: int
    latency_ms: float
    tool_sequence: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Fingerprint / quotient objects
# --------------------------------------------------------------------------- #
@dataclass
class BehaviorFingerprint:
    """Behavioural fingerprint of a prompt over the sentinel panel (Sec 2.2)."""
    prompt_id: str
    task_id: str
    normalized_answers: list[Optional[str]]
    format_validity: list[bool]
    refusal_flags: list[bool]
    parse_error_types: list[Optional[str]]
    output_length_buckets: list[int]
    token_counts: list[int]
    latency_buckets: list[int]
    tool_sequences: Optional[list[list[str]]] = None
    code_error_types: Optional[list[str]] = None
    json_schema_paths_present: Optional[list[list[str]]] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PhenotypeCell:
    """An equivalence class of prompts with similar black-box behaviour."""
    cell_id: str
    task_id: str
    prompt_ids: list[str]
    representative_prompt_id: str
    medoid_prompt_id: str
    size: int
    source_methods: dict[str, int] = field(default_factory=dict)
    fingerprint_centroid: dict = field(default_factory=dict)
    intra_cell_distance_mean: float = 0.0
    intra_cell_distance_max: float = 0.0
    labeled_eval_count: int = 0
    mean_selector_score: Optional[float] = None
    score_variance: Optional[float] = None
    uncertainty: Optional[float] = None
    novelty_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CellState:
    """Mutable optimiser state for a phenotype cell (Sec 2.5)."""
    cell_id: str
    prompt_ids: list[str]
    representative_prompt_id: str
    n_labeled_evals: int = 0
    sum_score: float = 0.0
    sum_score_sq: float = 0.0
    per_prompt_scores: dict[str, list[float]] = field(default_factory=dict)
    uncertainty_bonus: float = 0.0
    novelty_bonus: float = 0.0
    redundancy_penalty: float = 0.0
    estimated_cost: float = 0.0

    @property
    def mean_score(self) -> float:
        return self.sum_score / self.n_labeled_evals if self.n_labeled_evals else 0.0

    @property
    def score_variance(self) -> float:
        if self.n_labeled_evals < 2:
            return 0.0
        mean = self.mean_score
        return max(0.0, self.sum_score_sq / self.n_labeled_evals - mean * mean)

    def record(self, prompt_id: str, score: float) -> None:
        self.n_labeled_evals += 1
        self.sum_score += score
        self.sum_score_sq += score * score
        self.per_prompt_scores.setdefault(prompt_id, []).append(score)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean_score"] = self.mean_score
        d["score_variance"] = self.score_variance
        return d


# --------------------------------------------------------------------------- #
# Run-level objects
# --------------------------------------------------------------------------- #
@dataclass
class SelectorRun:
    """The result of one selector on one (task, budget, seed)."""
    run_id: str
    method_name: str
    task_id: str
    budget: int
    seed: int
    selected_prompt_id: str
    selected_cell_id: Optional[str] = None
    algorithm_visible_calls: int = 0
    proposal_calls: int = 0
    reflection_calls: int = 0
    sentinel_calls: int = 0
    selector_dev_calls: int = 0
    final_test_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    dollar_cost: float = 0.0
    wall_clock_latency_ms: float = 0.0
    held_out_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
