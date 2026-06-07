"""BehaviorFingerprintExtractor (Sec 3.5).

Turns a prompt's sentinel outputs into a BehaviorFingerprint. Bucketing of
length and latency keeps the fingerprint robust to small numeric noise.
"""
from __future__ import annotations

from ..data.datastructures import APITrace, BehaviorFingerprint, SentinelOutput
from .normalizers import OutputNormalizer


def bucket_output_length(tokens: int) -> int:
    edges = [8, 16, 32, 64, 128, 256]
    for i, e in enumerate(edges):
        if tokens <= e:
            return i
    return len(edges)


def bucket_latency(latency_ms: float) -> int:
    edges = [50, 100, 200, 400, 800, 1600]
    for i, e in enumerate(edges):
        if latency_ms <= e:
            return i
    return len(edges)


class BehaviorFingerprintExtractor:
    def __init__(self, task_normalizer: OutputNormalizer):
        self.task_normalizer = task_normalizer

    def parse_sentinel(self, trace: APITrace, sentinel_id: str) -> SentinelOutput:
        parsed = self.task_normalizer.parse(trace.output_text)
        return SentinelOutput(
            prompt_id=trace.prompt_id,
            task_id=trace.task_id,
            sentinel_id=sentinel_id,
            call_id=trace.call_id,
            raw_output=trace.output_text,
            normalized_answer=parsed.normalized_answer,
            format_valid=parsed.format_valid,
            refusal_flag=parsed.refusal_flag,
            parse_error_type=parsed.error_type,
            output_tokens=trace.output_tokens,
            latency_ms=trace.latency_ms,
            tool_sequence=parsed.tool_sequence,
        )

    def extract(self, prompt_id: str, sentinel_outputs: list[SentinelOutput]) -> BehaviorFingerprint:
        """sentinel_outputs must be ordered consistently across prompts."""
        return BehaviorFingerprint(
            prompt_id=prompt_id,
            task_id=sentinel_outputs[0].task_id,
            normalized_answers=[s.normalized_answer for s in sentinel_outputs],
            format_validity=[s.format_valid for s in sentinel_outputs],
            refusal_flags=[s.refusal_flag for s in sentinel_outputs],
            parse_error_types=[s.parse_error_type for s in sentinel_outputs],
            output_length_buckets=[bucket_output_length(s.output_tokens) for s in sentinel_outputs],
            token_counts=[s.output_tokens for s in sentinel_outputs],
            latency_buckets=[bucket_latency(s.latency_ms) for s in sentinel_outputs],
            tool_sequences=None,
        )
