"""Shared CLI arguments for the run scripts: target model, decoding, parallelism.

Add with ``add_runtime_args(parser)`` and build a ModelConfig with
``build_model_config(args, ...)``. Keeps --provider/--model/--temperature/
--max-output-tokens/--workers/--parallel-backend consistent across every runner.
"""
from __future__ import annotations

import argparse
import sys

from .api.executor import ModelConfig

# Canonical selector/control catalog (matches scripts.run_mvp.selector_registry).
SELECTOR_METHODS = [
    "pqpo", "score_only", "random_minibatch", "stratified_minibatch",
    "successive_halving", "hyperband", "asha", "bo_embedding",
    "lexical_cluster", "embedding_cluster", "score_bins", "random_quotient",
    "map_elites", "semantic_cache",
]
# Generator / source-method baselines (build candidate pools; also compared).
GENERATOR_METHODS = ["GEPA", "MIPROv2", "CAPO", "APE"]


def add_runtime_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = ap.add_argument_group("runtime (target model / decoding / parallelism)")
    g.add_argument("--provider", choices=["sim", "openai", "anthropic"], default=None,
                   help="target-model provider (default: sim for simulated runs, "
                        "else openai)")
    g.add_argument("--model", default=None,
                   help="target-model name, e.g. gpt-4o-mini / claude-3-5-haiku")
    g.add_argument("--temperature", type=float, default=0.0,
                   help="decoding temperature for target-model calls")
    g.add_argument("--max-output-tokens", type=int, default=256,
                   help="max output tokens per target-model call")
    g.add_argument("--workers", type=int, default=1,
                   help="parallel workers. 1 = sequential. Parallelises the "
                        "embarrassingly-parallel call phases (fingerprinting, "
                        "held-out scoring).")
    g.add_argument("--parallel-backend", choices=["thread", "process"], default="thread",
                   help="parallel backend for supported call phases. Use 'thread' "
                        "for real API providers; use 'process' for offline/sim "
                        "runs when you want true multiprocessing.")
    return ap


def add_method_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = ap.add_argument_group("methods")
    g.add_argument("--methods", nargs="*", default=None, metavar="M",
                   help="subset of methods to run (space- or comma-separated), e.g. "
                        "'--methods pqpo GEPA MIPROv2'. Default: all. "
                        "See --list-methods.")
    g.add_argument("--list-methods", action="store_true",
                   help="print the available methods for this runner and exit")
    return ap


def _parse_method_tokens(raw):
    if not raw:
        return None
    out = []
    for tok in raw:
        out += [t for t in tok.replace(",", " ").split() if t]
    return out or None


def resolve_methods(args, available, *, always_include=("pqpo",)):
    """Return the chosen subset of `available` (catalog order preserved).

    Names are validated against `available`; matching is case-insensitive for
    generator names (GEPA/gepa). `always_include` methods (e.g. pqpo, the
    comparison anchor) are added back if present in `available` and omitted."""
    requested = _parse_method_tokens(getattr(args, "methods", None))
    if requested is None:
        return list(available)
    lower = {m.lower(): m for m in available}
    chosen_set, unknown = set(), []
    for r in requested:
        if r in available:
            chosen_set.add(r)
        elif r.lower() in lower:
            chosen_set.add(lower[r.lower()])
        else:
            unknown.append(r)
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}\navailable: {list(available)}")
    chosen = [m for m in available if m in chosen_set]    # preserve catalog order
    for m in always_include:
        if m in available and m not in chosen:
            chosen.insert(0, m)
    return chosen


def maybe_list_methods(args, available):
    """If --list-methods was passed, print and exit."""
    if getattr(args, "list_methods", False):
        print("Available methods:")
        for m in available:
            print(f"  {m}")
        sys.exit(0)


def build_model_config(args, *, simulated: bool, sim_model: str = "sim-target",
                       default_provider: str = "openai",
                       default_model: str = "gpt-4o-mini") -> ModelConfig:
    """Construct a ModelConfig from parsed args.

    simulated=True forces provider='sim' (offline runs); otherwise --provider /
    --model (or the supplied defaults) select the billed target model."""
    if simulated:
        return ModelConfig(provider="sim", model_name=sim_model,
                           default_temperature=args.temperature,
                           default_max_output_tokens=args.max_output_tokens)
    provider = args.provider or default_provider
    if provider == "sim":          # user asked for sim provider on a real-data run
        return ModelConfig("sim", sim_model, default_temperature=args.temperature,
                           default_max_output_tokens=args.max_output_tokens)
    model = args.model or (default_model if provider == "openai"
                           else "claude-3-5-haiku")
    return ModelConfig(provider=provider, model_name=model,
                       default_temperature=args.temperature,
                       default_max_output_tokens=args.max_output_tokens)
