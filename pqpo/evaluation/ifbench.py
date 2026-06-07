"""IFBench / IFEval-style instruction-following with verifiable constraints.

Each item is a prompt plus a list of *programmatically verifiable* instructions
(e.g. "respond in all lowercase", "include the word X at least twice", "use
exactly 3 bullet points"). The behavioural signature of a prompt is the
per-constraint satisfaction vector across the sentinel panel — a rich, code-like
fingerprint that is a natural fit for the phenotype quotient.

Two paths:
  * REAL data (IFEval/IFBench via HuggingFace): real model text checked by the
    verifiers below. Score = strict prompt-level accuracy (all constraints met).
  * SYNTHETIC/offline: keyword-marker constraints that are independently
    satisfiable, with a deterministic SimIFTarget whose per-constraint
    satisfaction is profile-driven, so the whole pipeline runs without keys.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..data.datastructures import CandidatePrompt, TaskExample


# --------------------------------------------------------------------------- #
# Verifiers (IFEval-style). Each: (response, args) -> bool.
# --------------------------------------------------------------------------- #
def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


VERIFIERS: dict[str, Callable[[str, dict], bool]] = {
    "all_lowercase": lambda r, a: r.strip() != "" and r == r.lower() and any(c.isalpha() for c in r),
    "all_uppercase": lambda r, a: r.strip() != "" and r == r.upper() and any(c.isalpha() for c in r),
    "keyword": lambda r, a: len(re.findall(re.escape(a["word"]).lower(), r.lower())) >= a.get("min", 1),
    "forbidden_word": lambda r, a: a["word"].lower() not in r.lower(),
    "min_words": lambda r, a: len(_words(r)) >= a["n"],
    "max_words": lambda r, a: len(_words(r)) <= a["n"],
    "num_bullets": lambda r, a: sum(1 for ln in r.splitlines()
                                    if ln.strip().startswith(("*", "-"))) == a["n"],
    "end_phrase": lambda r, a: r.strip().endswith(a["phrase"]),
    "start_phrase": lambda r, a: r.strip().startswith(a["phrase"]),
    "num_sentences_min": lambda r, a: len(re.findall(r"[.!?]", r)) >= a["n"],
    "json_format": lambda r, a: _is_json(r),
    "title_case": lambda r, a: r.strip().istitle(),
}


def _is_json(r: str) -> bool:
    s, e = r.find("{"), r.rfind("}")
    if s == -1 or e == -1:
        return False
    try:
        json.loads(r[s:e + 1]); return True
    except Exception:
        return False


@dataclass
class IFItem:
    item_id: str
    prompt: str
    instructions: list[tuple[str, dict]]      # built-in [(verifier_id, args), ...]
    # Official IFBench/IFEval spec (used when the official registry is available):
    official_ids: list[str] = None
    official_kwargs: list[dict] = None
    metadata: dict = field(default_factory=dict)


def check_item(response: str, item: IFItem) -> list[bool]:
    """Per-constraint satisfaction. Uses the OFFICIAL IFBench verifiers when the
    item carries an official spec and the registry is importable; otherwise the
    built-in verifier subset (synthetic items, or approximate real scoring)."""
    if item.official_ids:
        from .ifbench_official import check_following_official, load_registry
        official = check_following_official(load_registry(), item.prompt,
                                            item.official_ids, item.official_kwargs,
                                            response)
        if official is not None:
            return official
        # registry unavailable: fall back to whatever built-in instructions exist
    out = []
    for vid, args in item.instructions:
        fn = VERIFIERS.get(vid)
        try:
            out.append(bool(fn(response, args)) if fn else False)
        except Exception:
            out.append(False)
    return out


# --------------------------------------------------------------------------- #
# Scorer (strict prompt-level accuracy) + example wrapping
# --------------------------------------------------------------------------- #
class IFScorer:
    def __init__(self, executor, strict: bool = True):
        self.executor = executor
        self.strict = strict

    def score_one(self, prompt: CandidatePrompt, example: TaskExample,
                  call_type: str = "selector_dev") -> float:
        item: IFItem = example.metadata["if_item"]
        trace = self.executor.run_prompt_on_example(prompt, example, call_type=call_type)
        sat = check_item(trace.output_text, item)
        if not sat:
            return 0.0
        return 1.0 if (all(sat) if self.strict else False) else (
            float(np.mean(sat)) if not self.strict else 0.0)

    def score_many(self, prompt, examples, call_type="selector_dev"):
        return [self.score_one(prompt, e, call_type) for e in examples]


def if_items_to_examples(items: list[IFItem], task_id: str = "ifbench") -> list[TaskExample]:
    return [TaskExample(it.item_id, task_id, it.prompt, None, {"if_item": it})
            for it in items]


# --------------------------------------------------------------------------- #
# Per-constraint behavioural fingerprint
# --------------------------------------------------------------------------- #
class IFBehaviorFingerprintExtractor:
    """Fingerprint = per-(item, constraint) satisfaction vector. Behavioural
    equivalence = satisfies the same constraints on the same sentinel items."""

    def extract(self, prompt: CandidatePrompt, sentinel_items: list[IFItem], executor):
        from ..data.datastructures import BehaviorFingerprint
        answers, fmt, refusal, errs, lenb, tok, lat = [], [], [], [], [], [], []
        for item in sentinel_items:
            ex = TaskExample(item.item_id, prompt.task_id, item.prompt, None,
                             {"if_item": item})
            trace = executor.run_prompt_on_example(prompt, ex, call_type="sentinel")
            sat = check_item(trace.output_text, item)
            for s in sat:
                answers.append("sat" if s else "unsat")
                fmt.append(True)
                refusal.append(trace.output_text.strip() == "")
                errs.append("sat" if s else "unsat")
                lenb.append(0); tok.append(0); lat.append(0)
        return BehaviorFingerprint(
            prompt_id=prompt.prompt_id, task_id=prompt.task_id,
            normalized_answers=answers, format_validity=fmt, refusal_flags=refusal,
            parse_error_types=errs, output_length_buckets=lenb,
            token_counts=tok, latency_buckets=lat)


# --------------------------------------------------------------------------- #
# Deterministic simulated IF target + synthetic items
# --------------------------------------------------------------------------- #
from ..api.sim_target import BehaviorProfile, _u01, approx_tokens, build_default_profiles  # noqa: E402


@dataclass
class SimIFTarget:
    """Emits a response that includes exactly the keyword markers it 'satisfies',
    with per-constraint satisfaction governed by the prompt's behaviour profile."""
    profiles: dict
    items_by_id: dict

    model_provider = "sim"
    model_name = "sim-if-target"
    model_version = "v1"

    def generate(self, prompt: CandidatePrompt, example: TaskExample,
                 temperature: float = 0.0, max_output_tokens: int = 256,
                 seed: Optional[int] = None):
        profile = self.profiles[prompt.metadata["behavior_profile_id"]]
        item = self.items_by_id[example.example_id]
        parts = []
        for ci, (vid, args) in enumerate(item.instructions):
            key = f"{profile.profile_id}|{item.item_id}|{ci}"
            if _u01("if", key) < profile.skill and vid == "keyword":
                parts.append(args["word"])              # satisfy by including marker
        text = "response " + " ".join(parts)
        in_tok = approx_tokens(prompt.prompt_text + example.input_text)
        return text, in_tok, approx_tokens(text), profile.latency_base


def build_synthetic_if_items(n_items: int, rng: np.random.Generator,
                             constraints_per_item: int = 4) -> list[IFItem]:
    items = []
    for i in range(n_items):
        instrs = []
        for c in range(constraints_per_item):
            marker = f"kw{int(rng.integers(0, 9999)):04d}"
            instrs.append(("keyword", {"word": marker, "min": 1}))
        items.append(IFItem(f"if_{i:04d}",
                            f"Follow all {constraints_per_item} instructions for item {i}.",
                            instrs))
    return items


def build_sim_if_target(n_profiles: int, n_items: int, rng: np.random.Generator,
                        constraints_per_item: int = 4):
    profiles = build_default_profiles(rng, n_profiles)
    items = build_synthetic_if_items(n_items, rng, constraints_per_item)
    return SimIFTarget(profiles, {it.item_id: it for it in items}), items


def load_ifbench(n_sentinel=12, n_dev=80, n_test=100, seed=0, source="hf",
                 dataset="allenai/IFBench_test"):
    """Real IFBench loader. Returns (sentinels, dev, test) of IFItem carrying the
    OFFICIAL (instruction_id_list, kwargs) spec, so scoring uses the official
    IFBench verifiers when the registry is importable (see ifbench_official).

    Falls back to synthetic items when datasets is unavailable. If requested split
    sizes exceed the dataset (IFBench_test has 300 items), they are capped and the
    remainder goes to the test split."""
    try:
        import datasets  # noqa
        have = source == "hf"
    except Exception:
        have = False
    if not have:
        items = build_synthetic_if_items(n_sentinel + n_dev + n_test + 20,
                                         np.random.default_rng(seed))
    else:
        from .ifbench_official import official_available
        if not official_available():
            raise SystemExit(
                "Full IFBench scoring requires the OFFICIAL IFBench verifiers, which "
                "are not importable. Its out-of-domain constraints (e.g. "
                "'words:odd_even_syllables') have no built-in coverage, so every item "
                "would score 0.\n\nOne-time setup:\n"
                "  pip install datasets\n"
                "  git clone https://github.com/allenai/IFBench\n"
                "  pip install -r IFBench/requirements.txt\n"
                "  export IFBENCH_DIR=/abs/path/to/IFBench   "
                "# the REPO ROOT (where instructions_registry.py lives)\n\n"
                "Then re-run. (The simulated path, --source sim, needs none of this.)")
        from datasets import load_dataset
        ds = load_dataset(dataset, split="train")
        items = []
        for i, r in enumerate(ds):
            ids = list(r.get("instruction_id_list", []))
            kw = list(r.get("kwargs", []))
            # built-in fallback mapping (best-effort, partial coverage)
            builtin = _map_ifeval_instructions(ids, kw)
            items.append(IFItem(item_id=r.get("key", f"ifbench_{i}"),
                                prompt=r["prompt"], instructions=builtin,
                                official_ids=ids, official_kwargs=kw))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    n = len(items)
    n_sentinel = min(n_sentinel, n)
    n_dev = min(n_dev, max(0, n - n_sentinel))
    n_test = max(0, n - n_sentinel - n_dev) if (n_sentinel + n_dev + n_test) > n else n_test
    g = lambda a, b: [items[j] for j in idx[a:b]]
    return (g(0, n_sentinel), g(n_sentinel, n_sentinel + n_dev),
            g(n_sentinel + n_dev, n_sentinel + n_dev + n_test))


def _map_ifeval_instructions(id_list, kwargs_list):
    """Best-effort map IFEval instruction ids -> our verifiers (extend as needed)."""
    mapping = {
        "change_case:english_lowercase": ("all_lowercase", {}),
        "change_case:english_capital": ("all_uppercase", {}),
        "keywords:existence": ("keyword", None),       # word from kwargs
        "length_constraints:number_words": ("min_words", None),
        "detectable_format:number_bullet_lists": ("num_bullets", None),
        "startend:end_checker": ("end_phrase", None),
    }
    out = []
    for iid, kw in zip(id_list, kwargs_list or []):
        if iid not in mapping:
            continue
        vid, args = mapping[iid]
        if args is None:                                # pull args from kwargs
            kw = kw or {}
            if vid == "keyword" and kw.get("keywords"):
                args = {"word": kw["keywords"][0], "min": 1}
            elif vid == "min_words" and kw.get("num_words"):
                args = {"n": int(kw["num_words"])}
            elif vid == "num_bullets" and kw.get("num_bullets"):
                args = {"n": int(kw["num_bullets"])}
            elif vid == "end_phrase" and kw.get("end_phrase"):
                args = {"phrase": kw["end_phrase"]}
            else:
                args = {}
        out.append((vid, args))
    return out
