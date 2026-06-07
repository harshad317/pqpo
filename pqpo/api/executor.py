"""APIExecutor: the single choke point for all target-model calls (Sec 3.3).

Responsibilities
  1. Deterministic request hashing + local cache (replayable experiments).
  2. Complete token/cost logging via the CostLedger.
  3. Retry handling for real providers.
  4. Strict separation of call types (proposal/reflection/sentinel/...).
  5. A pluggable backend: the simulated target, OpenAI, or Anthropic.

Wiring real keys
  Set ``provider`` to "openai" or "anthropic" in the ModelConfig and export
  OPENAI_API_KEY / ANTHROPIC_API_KEY. The OpenAI/Anthropic SDK calls are guarded
  imports so the package runs with zero external deps in simulated mode.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..data.datastructures import APITrace, CandidatePrompt, TaskExample
from ..logging_utils.cost_ledger import CostLedger
from . import pricing
from .cache import APICache, request_hash, text_hash
from .sim_target import SimTarget, approx_tokens


@dataclass
class ModelConfig:
    provider: str = "sim"                 # sim | openai | anthropic
    model_name: str = "sim-target"
    model_version: Optional[str] = "v1"
    default_temperature: float = 0.0
    default_max_output_tokens: int = 256
    system_prompt_is_candidate: bool = True   # candidate prompt used as system msg
    extra: dict = field(default_factory=dict)

    @property
    def model_key(self) -> str:
        return f"{self.provider}/{self.model_name}"


class APIExecutor:
    def __init__(
        self,
        model_config: ModelConfig,
        cost_ledger: CostLedger,
        cache: Optional[APICache] = None,
        run_id: str = "run",
        sim_target: Optional[SimTarget] = None,
        max_retries: int = 3,
    ):
        self.cfg = model_config
        self.ledger = cost_ledger
        self.cache = cache or APICache(None)
        self.run_id = run_id
        self.sim = sim_target
        self.max_retries = max_retries
        if self.cfg.provider == "sim" and self.sim is None:
            raise ValueError("provider=sim requires a SimTarget instance")

    # -- public api --------------------------------------------------------- #
    def run_prompt_on_example(
        self,
        prompt: CandidatePrompt,
        example: TaskExample,
        call_type: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> APITrace:
        temperature = self.cfg.default_temperature if temperature is None else temperature
        max_output_tokens = max_output_tokens or self.cfg.default_max_output_tokens
        full_input = self._compose_input(prompt.prompt_text, example.input_text)
        rhash = request_hash(
            self.cfg.model_key, self.cfg.model_version, full_input,
            temperature, max_output_tokens, seed,
        )
        rid = run_id or self.run_id

        cached = self.cache.get(rhash)
        if cached is not None:
            trace = self._build_trace(
                prompt, example, call_type, full_input, rhash, rid,
                output_text=cached["output_text"],
                input_tokens=cached["input_tokens"],
                output_tokens=cached["output_tokens"],
                latency_ms=cached["latency_ms"],
                temperature=temperature, max_output_tokens=max_output_tokens,
                seed=seed, cache_hit=True,
            )
            self.ledger.record_api_call(trace)
            return trace

        output_text, in_tok, out_tok, latency = self._backend_call(
            prompt, example, full_input, temperature, max_output_tokens, seed
        )
        self.cache.put(rhash, {
            "output_text": output_text, "input_tokens": in_tok,
            "output_tokens": out_tok, "latency_ms": latency,
        })
        trace = self._build_trace(
            prompt, example, call_type, full_input, rhash, rid,
            output_text=output_text, input_tokens=in_tok, output_tokens=out_tok,
            latency_ms=latency, temperature=temperature,
            max_output_tokens=max_output_tokens, seed=seed, cache_hit=False,
        )
        self.ledger.record_api_call(trace)
        return trace

    def batch_run(
        self,
        prompts: list[CandidatePrompt],
        examples: list[TaskExample],
        call_type: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list[APITrace]:
        out = []
        for p in prompts:
            for ex in examples:
                out.append(self.run_prompt_on_example(
                    p, ex, call_type, temperature, max_output_tokens, seed))
        return out

    def log_call_cost(self, trace: APITrace) -> None:
        self.ledger.record_api_call(trace)

    # -- generic completion for proposal/reflection/meta calls -------------- #
    def complete(self, system_text: str, user_text: str, call_type: str,
                 task_id: str = "meta", temperature: float = 0.8,
                 max_output_tokens: int = 512, seed: Optional[int] = None,
                 run_id: Optional[str] = None) -> APITrace:
        """A single proposal/reflection LLM call (real providers).

        Used by the candidate generators (GEPA/MIPROv2/CAPO) to produce or revise
        prompts. Logged with the supplied call_type so proposal/reflection cost is
        accounted separately (Sec 5.4). For provider='sim' the generators supply
        text directly via the SimProposalLLM and log cost with
        ``record_meta_call`` instead — this method is for billed providers."""
        full_input = f"<SYSTEM>\n{system_text}\n</SYSTEM>\n<USER>\n{user_text}\n</USER>"
        rhash = request_hash(self.cfg.model_key, self.cfg.model_version,
                             full_input, temperature, max_output_tokens, seed)
        cached = self.cache.get(rhash)
        if cached is not None:
            tr = self._build_meta_trace(call_type, task_id, full_input, rhash,
                                        run_id or self.run_id, cached["output_text"],
                                        cached["input_tokens"], cached["output_tokens"],
                                        cached["latency_ms"], temperature,
                                        max_output_tokens, seed, cache_hit=True)
            self.ledger.record_api_call(tr)
            return tr
        if self.cfg.provider == "openai":
            text, it, ot, lat = self._openai_complete(system_text, user_text,
                                                       temperature, max_output_tokens, seed)
        elif self.cfg.provider == "anthropic":
            text, it, ot, lat = self._anthropic_complete(system_text, user_text,
                                                          temperature, max_output_tokens, seed)
        else:
            raise ValueError("complete() is for billed providers; use SimProposalLLM "
                             "for provider='sim'")
        self.cache.put(rhash, {"output_text": text, "input_tokens": it,
                               "output_tokens": ot, "latency_ms": lat})
        tr = self._build_meta_trace(call_type, task_id, full_input, rhash,
                                    run_id or self.run_id, text, it, ot, lat,
                                    temperature, max_output_tokens, seed, cache_hit=False)
        self.ledger.record_api_call(tr)
        return tr

    def record_meta_call(self, call_type: str, task_id: str, input_text: str,
                         output_text: str, input_tokens: int, output_tokens: int,
                         latency_ms: float, run_id: Optional[str] = None,
                         prompt_id: Optional[str] = None) -> APITrace:
        """Record a (simulated) proposal/reflection call's cost in the ledger."""
        rhash = text_hash(input_text + "->" + output_text)
        tr = self._build_meta_trace(call_type, task_id, input_text, rhash,
                                    run_id or self.run_id, output_text,
                                    input_tokens, output_tokens, latency_ms,
                                    temperature=0.8, max_output_tokens=0, seed=None,
                                    cache_hit=False, prompt_id=prompt_id)
        self.ledger.record_api_call(tr)
        return tr

    def _build_meta_trace(self, call_type, task_id, full_input, rhash, rid,
                          output_text, input_tokens, output_tokens, latency_ms,
                          temperature, max_output_tokens, seed, cache_hit,
                          prompt_id=None) -> APITrace:
        cost = pricing.dollar_cost(self.cfg.model_key, input_tokens, output_tokens)
        return APITrace(
            call_id=str(uuid.uuid4()), run_id=rid,
            model_provider=self.cfg.provider, model_name=self.cfg.model_name,
            model_version=self.cfg.model_version, task_id=task_id,
            call_type=call_type, input_text_hash=text_hash(full_input),
            full_input_text=full_input, output_text=output_text,
            temperature=temperature, max_output_tokens=max_output_tokens,
            input_tokens=input_tokens, output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens, latency_ms=latency_ms,
            dollar_cost=cost, cache_hit=cache_hit, request_hash=rhash,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            prompt_id=prompt_id, example_id=None, seed=seed)

    def _openai_complete(self, system_text, user_text, temperature, max_tokens, seed):
        from openai import OpenAI  # pragma: no cover
        client = OpenAI(); t0 = time.time()
        resp = client.chat.completions.create(
            model=self.cfg.model_name,
            messages=[{"role": "system", "content": system_text},
                      {"role": "user", "content": user_text}],
            temperature=temperature, max_tokens=max_tokens, seed=seed)
        lat = (time.time() - t0) * 1000.0
        return (resp.choices[0].message.content or "",
                resp.usage.prompt_tokens, resp.usage.completion_tokens, lat)

    def _anthropic_complete(self, system_text, user_text, temperature, max_tokens, seed):
        import anthropic  # pragma: no cover
        client = anthropic.Anthropic(); t0 = time.time()
        resp = client.messages.create(
            model=self.cfg.model_name, system=system_text,
            messages=[{"role": "user", "content": user_text}],
            temperature=temperature, max_tokens=max_tokens)
        lat = (time.time() - t0) * 1000.0
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens, lat

    # -- internals ---------------------------------------------------------- #
    def _compose_input(self, prompt_text: str, example_input: str) -> str:
        if self.cfg.system_prompt_is_candidate:
            return f"<SYSTEM>\n{prompt_text}\n</SYSTEM>\n<USER>\n{example_input}\n</USER>"
        return f"{prompt_text}\n\n{example_input}"

    def _build_trace(self, prompt, example, call_type, full_input, rhash, rid,
                     output_text, input_tokens, output_tokens, latency_ms,
                     temperature, max_output_tokens, seed, cache_hit) -> APITrace:
        cost = pricing.dollar_cost(self.cfg.model_key, input_tokens, output_tokens)
        return APITrace(
            call_id=str(uuid.uuid4()),
            run_id=rid,
            model_provider=self.cfg.provider,
            model_name=self.cfg.model_name,
            model_version=self.cfg.model_version,
            task_id=example.task_id,
            call_type=call_type,
            input_text_hash=text_hash(full_input),
            full_input_text=full_input,
            output_text=output_text,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            dollar_cost=cost,
            cache_hit=cache_hit,
            request_hash=rhash,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            prompt_id=prompt.prompt_id,
            example_id=example.example_id,
            seed=seed,
        )

    def _backend_call(self, prompt, example, full_input, temperature,
                      max_output_tokens, seed) -> tuple[str, int, int, float]:
        if self.cfg.provider == "sim":
            return self.sim.generate(prompt, example, temperature, max_output_tokens, seed)
        if self.cfg.provider == "openai":
            return self._openai_call(prompt, example, temperature, max_output_tokens, seed)
        if self.cfg.provider == "anthropic":
            return self._anthropic_call(prompt, example, temperature, max_output_tokens, seed)
        raise ValueError(f"unknown provider {self.cfg.provider}")

    def _openai_call(self, prompt, example, temperature, max_output_tokens, seed):
        from openai import OpenAI  # guarded import
        client = OpenAI()
        t0 = time.time()
        last_err = None
        for _ in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.cfg.model_name,
                    messages=[
                        {"role": "system", "content": prompt.prompt_text},
                        {"role": "user", "content": example.input_text},
                    ],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    seed=seed,
                )
                latency = (time.time() - t0) * 1000.0
                msg = resp.choices[0].message.content or ""
                usage = resp.usage
                return msg, usage.prompt_tokens, usage.completion_tokens, latency
            except Exception as e:  # pragma: no cover - network path
                last_err = e
                time.sleep(1.0)
        raise RuntimeError(f"OpenAI call failed after retries: {last_err}")

    def _anthropic_call(self, prompt, example, temperature, max_output_tokens, seed):
        import anthropic  # guarded import
        client = anthropic.Anthropic()
        t0 = time.time()
        last_err = None
        for _ in range(self.max_retries):
            try:
                resp = client.messages.create(
                    model=self.cfg.model_name,
                    system=prompt.prompt_text,
                    messages=[{"role": "user", "content": example.input_text}],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                latency = (time.time() - t0) * 1000.0
                text = "".join(b.text for b in resp.content if b.type == "text")
                return text, resp.usage.input_tokens, resp.usage.output_tokens, latency
            except Exception as e:  # pragma: no cover - network path
                last_err = e
                time.sleep(1.0)
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")
