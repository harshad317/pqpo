"""Per-task output normalizers (Sec 3.6).

A normalizer maps a raw model output to a ParsedOutput: a canonical answer,
format validity, refusal flag, and an error type. One normalizer per task family.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ..data.datastructures import ParsedOutput

_REFUSAL_PATTERNS = [
    "i'm sorry", "i am sorry", "i can't help", "i cannot help",
    "i can't assist", "cannot assist", "i won't", "as an ai",
]


def detect_refusal(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _REFUSAL_PATTERNS)


class OutputNormalizer:
    def parse(self, output_text: str) -> ParsedOutput:  # pragma: no cover
        raise NotImplementedError


class ClassificationNormalizer(OutputNormalizer):
    """Extracts the first valid label. Requires an ``Answer: <label>`` style
    response for format validity, but will still recover a label if present."""

    def __init__(self, label_set: list[str]):
        self.label_set = [l.lower() for l in label_set]
        self._orig = {l.lower(): l for l in label_set}

    def parse(self, output_text: str) -> ParsedOutput:
        text = output_text.strip()
        low = text.lower()
        if detect_refusal(low):
            return ParsedOutput(None, False, True, "refusal")
        # Strict format: "answer: X"
        m = re.search(r"answer\s*[:\-]\s*([a-z0-9_\-]+)", low)
        strict_label = None
        if m and m.group(1) in self.label_set:
            strict_label = self._orig[m.group(1)]
        # Loose recovery: first label token anywhere.
        loose_label = None
        for lab in self.label_set:
            if re.search(rf"\b{re.escape(lab)}\b", low):
                loose_label = self._orig[lab]
                break
        label = strict_label or loose_label
        format_valid = strict_label is not None
        return ParsedOutput(
            normalized_answer=label,
            format_valid=format_valid,
            refusal_flag=False,
            error_type=None if label else "no_valid_label",
        )


class ReasoningNormalizer(OutputNormalizer):
    """Extracts a final numeric or multiple-choice answer."""

    def __init__(self, choice_set: Optional[list[str]] = None):
        self.choice_set = [c.lower() for c in choice_set] if choice_set else None

    def parse(self, output_text: str) -> ParsedOutput:
        if detect_refusal(output_text):
            return ParsedOutput(None, False, True, "refusal")
        low = output_text.lower()
        m = re.search(r"answer\s*[:\-]\s*([a-z0-9_\.\-]+)", low)
        ans = m.group(1) if m else None
        if ans is None:
            nums = re.findall(r"-?\d+(?:\.\d+)?", output_text)
            ans = nums[-1] if nums else None
        format_valid = m is not None
        norm = self._normalize_numeric(ans) if ans is not None else None
        return ParsedOutput(
            normalized_answer=norm,
            format_valid=format_valid,
            refusal_flag=False,
            error_type=None if norm is not None else "missing_final_answer",
        )

    @staticmethod
    def _normalize_numeric(ans: str) -> str:
        try:
            f = float(ans)
            return str(int(f)) if f.is_integer() else str(f)
        except ValueError:
            return ans.strip().lower()


class Banking77Normalizer(OutputNormalizer):
    """High-cardinality (77 intents) classifier normalizer.

    Intents are snake_case (e.g. 'card_arrival'). Accepts the intent token with or
    without underscores/spaces, and an optional 'Answer:' prefix."""

    def __init__(self, label_set: list[str]):
        self.labels = label_set
        # map normalized variants -> canonical label
        self._lookup = {}
        for lab in label_set:
            for v in {lab.lower(), lab.lower().replace("_", " "),
                      lab.lower().replace("_", "")}:
                self._lookup[v] = lab

    def parse(self, output_text: str) -> ParsedOutput:
        if detect_refusal(output_text):
            return ParsedOutput(None, False, True, "refusal")
        text = output_text.strip().lower()
        m = re.search(r"answer\s*[:\-]\s*(.+)", text)
        cand = (m.group(1) if m else text).strip().strip(".\"' ")
        # try full-string, then collapsed, then token windows
        for key in (cand, cand.replace(" ", "_"), cand.replace(" ", "")):
            if key in self._lookup:
                return ParsedOutput(self._lookup[key], m is not None, False, None)
        for lab in self.labels:
            if re.search(rf"\b{re.escape(lab.lower())}\b", text) or \
               lab.lower().replace("_", " ") in text:
                return ParsedOutput(lab, m is not None, False, None)
        return ParsedOutput(None, False, False, "no_valid_label")


class GSM8KNormalizer(OutputNormalizer):
    """Numeric-answer normalizer for grade-school math. Prefers the GSM8K
    '#### <n>' convention, else the last number in the text."""

    def parse(self, output_text: str) -> ParsedOutput:
        if detect_refusal(output_text):
            return ParsedOutput(None, False, True, "refusal")
        m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", output_text)
        strict = m is not None
        if not m:
            m2 = re.search(r"answer\s*[:\-]?\s*(-?[\d,]+(?:\.\d+)?)", output_text.lower())
            m = m2
        num = None
        if m:
            num = m.group(1)
        else:
            allnums = re.findall(r"-?[\d,]+(?:\.\d+)?", output_text)
            num = allnums[-1] if allnums else None
        if num is None:
            return ParsedOutput(None, False, False, "missing_final_answer")
        canon = num.replace(",", "")
        try:
            f = float(canon)
            canon = str(int(f)) if f.is_integer() else str(f)
        except ValueError:
            pass
        return ParsedOutput(canon, strict, False, None if strict else "no_marker")


class HoVerNormalizer(OutputNormalizer):
    """Claim-verification label normalizer: SUPPORTED vs NOT_SUPPORTED."""

    _SUP = ["supported", "support", "true", "correct", "yes", "entailed"]
    _NOT = ["not_supported", "not supported", "refuted", "false", "incorrect",
            "no", "unsupported", "contradicted"]

    def parse(self, output_text: str) -> ParsedOutput:
        if detect_refusal(output_text):
            return ParsedOutput(None, False, True, "refusal")
        low = output_text.strip().lower()
        # check NOT first (it contains 'support' as a substring)
        if any(k in low for k in self._NOT):
            return ParsedOutput("NOT_SUPPORTED", "answer" in low, False, None)
        if any(re.search(rf"\b{k}\b", low) for k in self._SUP):
            return ParsedOutput("SUPPORTED", "answer" in low, False, None)
        return ParsedOutput(None, False, False, "no_valid_label")


class JSONExtractionNormalizer(OutputNormalizer):
    """Parses JSON and validates required keys."""

    def __init__(self, required_keys: list[str]):
        self.required_keys = required_keys

    def parse(self, output_text: str) -> ParsedOutput:
        if detect_refusal(output_text):
            return ParsedOutput(None, False, True, "refusal")
        try:
            obj = self._extract_json(output_text)
            schema_valid = all(k in obj for k in self.required_keys)
            answer = json.dumps(obj, sort_keys=True)
            error = None if schema_valid else "schema_invalid"
        except Exception:
            return ParsedOutput(None, False, False, "json_parse_error")
        return ParsedOutput(answer, schema_valid, False, error)

    @staticmethod
    def _extract_json(text: str) -> dict:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no json object")
        return json.loads(text[start: end + 1])
