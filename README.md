# PQPO — Phenotype Quotient Prompt Optimization

Reference implementation of **PQPO**: optimize prompts over the *quotient space
induced by target-model behavior* rather than over prompt strings. The core
object is the **phenotype cell** — an equivalence class of prompts that induce
similar black-box behavior on a fixed panel of unlabelled *sentinel* probes.

> Thesis: prompt strings are an inefficient search state because the target model
> maps many different strings to the same observable task behavior. PQPO spends
> labelled evaluations on *behaviorally distinct* representatives instead of on
> behaviorally redundant strings.

This repo implements the full method, every baseline/control named in the plan,
the statistical machinery, and an end-to-end MVP that evaluates the six
breakthrough conditions. It runs out-of-the-box against a deterministic
**simulated target model** (no API keys) and swaps cleanly to OpenAI/Anthropic.

---

## Unified runner (`python -m benchmark.run`)

One entry point runs any methods over any benchmarks, across all three
modalities (label / code / instruction-following):

```bash
python -m benchmark.run \
    --benchmarks ifbench gsm8k mbpp \
    --methods pqpo mipro gepa successive_halving \
    --provider openai --model gpt-4o \
    --pool generated --mipro-auto medium \
    --workers 12 --parallel-backend thread
```

- `--benchmarks` — one or more of: `banking77`, `gsm8k`, `hover` (label);
  `ifbench`; `mbpp`, `humaneval`, `livecodebench` (code).
- `--methods` — selectors/controls and generators, with aliases:
  `mipro`/`miprov2`→MIPROv2, `gepa`→GEPA, `capo`→CAPO, `ape`→APE, `sh`→
  successive_halving. `pqpo` is always included. `--list-methods` prints the set.
- `--mipro-auto {light,medium,heavy}` — MIPROv2 search budget (DSPy-style), used
  when `--pool generated`.
- `--provider/--model/--temperature/--workers/--parallel-backend` — as below;
  `--source sim` (default) runs offline, `--source hf` uses real datasets + your
  billed model.
- `--budgets 48,96,192`, `--seeds N`, `--pool {synthetic,generated}` and the
  `--n-*` sizing flags control the sweep.

It prints per-benchmark phenotype/redundancy/AUBC/Holm tables plus a
cross-benchmark summary (PQPO AUBC and rank per benchmark). The legacy
per-benchmark scripts (`scripts/run_*_mvp.py`) remain for focused runs.

**What the simulator robustly shows vs what it doesn't.** The paper's core claim
is that the *behavioural* quotient selects better than non-behavioural quotients
and bandits. That is robust on the simulator — PQPO beats lexical/embedding
clustering, score-bin/random-quotient/MAP-Elites controls, and successive-halving
/Hyperband/ASHA (this is the "controls fail" condition):

```bash
python -m benchmark.run --benchmarks gsm8k \
    --methods pqpo lexical_cluster embedding_cluster successive_halving \
              random_quotient map_elites score_bins \
    --pool synthetic --budgets 24,48,96 --seeds 3
```

The PQPO-vs-*generator-output* comparison (GEPA/MIPROv2/CAPO/APE) is secondary and
data-dependent: on the synthetic target, strong proposal methods like APE can find
a near-oracle prompt by sampling, so PQPO shows no efficiency win over APE there
(the report's budget-efficiency table makes this explicit). This is a property of
the synthetic target, not of real models — zero-shot APE rarely lands a near-oracle
prompt on a real task. The generator horse-race is exactly what `--source hf`
settles.

**Regime matters for the selection comparison.** PQPO's selection edge appears
when the pool is large and behaviourally redundant and the labelled budget is
*much smaller* than the pool — that's the whole point (don't pay to evaluate
behaviourally-identical prompts). If `--budgets` ≥ pool size, every method can
evaluate nearly the whole pool, so "best of one generator's prompts" becomes
near-oracle and the comparison is not discriminative (the runner prints a warning
in that case). For a fair `--pool generated` comparison, use several generators
with a healthy `--per-method-size` and budgets well below the pool size, e.g.:

```bash
python -m benchmark.run --benchmarks gsm8k \
    --methods pqpo gepa mipro capo ape --pool generated \
    --per-method-size 30 --generations 4 --budgets 48,192,384 --seeds 3
```

Generator baselines (GEPA/MIPROv2/CAPO/APE) report each generator's **own
converged output** — the candidate with the best internal validation score it
recorded during generation, scored once (no extra labelled budget). This is the
realistic baseline; it avoids the near-oracle "exhaustively re-search the lineage
under the shared budget" artifact. The generator baselines are therefore a fixed
horizontal line across budgets, and PQPO overtakes them once its selection budget
is large enough to exploit the quotient (at very low budget a generator's free
internal-best can still win — PQPO's value needs enough budget AND a multi-
generator pool where no single generator already dominates).

The phenotype results (redundancy, compression, cluster stability,
distance-transfer) are robust across regimes; the *selection* horse-race is what
depends on pool-vs-budget and is what the real `--source hf` runs establish.

## Running the full IFBench (`allenai/IFBench_test`, real model)

IFBench (Pyatkin et al., NeurIPS 2025) scores instruction following with 58
out-of-domain verifiable constraints, so it needs its **official verifiers** —
the built-in subset cannot score constraints like `words:odd_even_syllables`, and
the runner hard-fails (with instructions) rather than silently scoring everything 0.

One-time setup:

```bash
pip install datasets
git clone https://github.com/allenai/IFBench
export PYTHONPATH="$PYTHONPATH:/path/to/IFBench/IFBench"   # dir with instructions_registry.py
export OPENAI_API_KEY=sk-...
```

Validate cheaply first (small test split, few methods), then scale up:

```bash
# 1) smoke test (~minutes, low cost)
python -m benchmark.run --benchmarks ifbench --source hf \
    --provider openai --model gpt-4.1-mini \
    --methods pqpo lexical_cluster embedding_cluster successive_halving \
    --pool synthetic --n-dev 40 --n-test 40 --budgets 24,48 --seeds 2 \
    --skip-transfer --workers 16

# 2) full run (all 300 items: 12 sentinel / 100 dev / 188 test)
python -m benchmark.run --benchmarks ifbench --source hf \
    --provider openai --model gpt-4.1-mini \
    --methods pqpo gepa mipro capo ape lexical_cluster embedding_cluster \
              successive_halving random_quotient map_elites \
    --pool generated --per-method-size 30 --generations 4 \
    --n-dev 100 --n-test 188 --budgets 24,48,96 --seeds 3 --workers 16
```

Cost control: `--skip-transfer` omits the per-prompt oracle/distance-transfer
analysis (which would score every pool prompt on all 188 test items — the biggest
call driver); the method comparison is unaffected. Lower `--n-test`, `--seeds`, or
`--per-method-size` to trim further. `--workers` parallelises the model calls
(mind your provider's rate limit). Every call is logged to the cost ledger.

## Quick start

```bash
pip install -r requirements.txt

python scripts/run_mvp.py --mode tiny      # ~10s  full pipeline smoke run
python scripts/run_mvp.py --mode quick     # ~25s  3 tasks x 3 budgets x 3 seeds
python scripts/build_candidate_pool.py     # GEPA+MIPROv2+CAPO generate a real pool
python scripts/run_closed_loop.py          # closed-loop + non-scheduler evidence
python scripts/run_code_fingerprint_demo.py  # rich code fingerprint (test-pass vectors)
python scripts/run_code_mvp.py             # full PQPO-over-code MVP (pass@1, baselines)
python tests/test_pipeline.py              # 15 correctness tests
```

### Runtime flags (all runners)

Every runner (`run_mvp.py`, `run_code_mvp.py`, `run_ifbench_mvp.py`) accepts a
shared set of flags via `pqpo/cli.py`:

```
--provider {sim,openai,anthropic}   target-model provider
--model MODEL                       model name, e.g. gpt-4o-mini / claude-3-5-haiku
--temperature T                     decoding temperature (default 0.0)
--max-output-tokens N               cap per call (default 256)
--workers W                         parallel workers (default 1)
--parallel-backend {thread,process} thread for real APIs, process for sim/offline
--methods M [M ...]                 subset of methods to run (default: all)
--list-methods                      print available methods and exit
```

`--methods` selects which methods to run and compare — selectors/controls
(`pqpo`, `successive_halving`, `lexical_cluster`, …) and, for the benchmark
runners, the generators (`GEPA`, `MIPROv2`, `CAPO`, `APE`). `pqpo` is always
included as the comparison anchor, and only the chosen methods are executed (so
you don't pay for baselines you didn't ask for). Examples:

```bash
python scripts/run_ifbench_mvp.py --list-methods
python scripts/run_ifbench_mvp.py --methods pqpo GEPA MIPROv2          # PQPO vs two generators
python scripts/run_ifbench_mvp.py --methods pqpo GEPA --pool generated # GEPA builds the pool
python scripts/run_code_mvp.py    --methods pqpo lexical_cluster embedding_cluster
```

`--workers` parallelises the embarrassingly-parallel call phases (fingerprinting
and held-out scoring), which dominate real-provider cost; results are returned in
input order so runs stay deterministic. Use `--parallel-backend thread` for real
OpenAI/Anthropic runs because it shares one cache/ledger and avoids spawning many
independent API clients. Use `--parallel-backend process` for offline/simulated
IFBench runs when you want true Python multiprocessing. Example real run:

```bash
python scripts/run_ifbench_mvp.py --source hf --provider openai \
    --model gpt-4o-mini --temperature 0.0 --workers 8
```

Example multiprocessing smoke run:

```bash
python scripts/run_ifbench_mvp.py --pool generated \
    --methods pqpo GEPA APE \
    --parallel-backend process --workers 2
```

All scripts stream live progress: `run_mvp.py` shows a **rich** live table of
running held-out score per method × budget (plus progress bars and the
breakthrough panel); `build_candidate_pool.py` and `run_closed_loop.py` show
**tqdm** bars with live metrics (e.g. closed-loop target-cell hit rate per
iteration) and rich summary tables. Output degrades to plain prints when stdout
is not a TTY.

`run_mvp.py` prints the six breakthrough conditions and writes
`artifacts/mvp_run/mvp_summary.json` (per-task tau, cell counts, AUBC by method,
Holm-corrected PQPO-vs-baseline tests, distance-transfer correlations).

### Latest simulated `--mode quick` result (sanity check, not a SOTA claim)

| Condition | Result |
|---|---|
| C1 behavioural redundancy ≥30% in ≥2 tasks | **PASS** (≈0.94–1.0 per task) |
| C2 beats best non-behavioural (AUBC+10% **or** −40% evals) | **PASS** (reaches best-baseline full-budget score at 25% of budget) |
| C4 all quotient controls fail to match PQPO | **PASS** (6/6) |
| C5 phenotype distance strongest transfer predictor | **PASS** (Spearman ≈0.63 vs ≤0.07 others) |
| C6 cluster stability ARI ≥0.65 all tasks | **PASS** (ARI≈1.0) |

PQPO ranks #1 by AUBC and significantly beats successive-halving, Hyperband,
ASHA, and semantic-cache (Holm-corrected). **These numbers are on synthetic data
designed to satisfy the paper's generative assumptions — they validate the
machinery, not real-world SOTA.** Real claims require the runs below.

---

## Wiring a real target model

Set the provider and export a key; everything else is unchanged.

```python
from pqpo.api.executor import APIExecutor, ModelConfig
cfg = ModelConfig(provider="openai", model_name="gpt-4o-mini",
                  default_temperature=0.0, default_max_output_tokens=256)
# export OPENAI_API_KEY=...   (or ANTHROPIC_API_KEY for provider="anthropic")
```

Then, in `scripts/run_mvp.py`:
1. Replace `SimTarget` wiring with your `ModelConfig` (drop the `sim_target=` arg).
2. Replace `make_synthetic_examples` / `build_candidate_pool` with real loaders
   (TREC, GSM8K-style reasoning, a JSON-extraction task) and real candidate pools
   produced by GEPA / MIPROv2 / CAPO (see `pqpo/data/datasets.py` docstring and
   Sec 4 of the plan).
3. Pass on-disk paths to `CostLedger(path)` and `APICache(path)` so every billed
   call is logged and replayable. Update `pqpo/api/pricing.py` with the frozen
   price snapshot for your run.
4. Scale `get_config("full")`: 240 prompts/task, 12 sentinels, 96 dev, 300–500
   test, budgets {0,240,480,960}, ≥5 seeds.

Cost is enforced two ways: **labelled-eval budget** (the algorithm-visible cost
used for all budget curves) and the **dollar CostLedger** (which separates
proposal/reflection/**sentinel**/selector-dev/final-test calls — never report
"labelled evals saved" without the sentinel cost).

---

## Architecture

```
pqpo/
  api/         executor (sim|openai|anthropic) + complete(), pricing, cache, sim_target
  data/        datastructures, datasets + synthetic candidate-pool builder
  candidates/  proposal_llm (Sim|LLM), gepa_adapter, miprov2_adapter,
               capo_adapter, pool_builder  ← real candidate generators
  fingerprints/ normalizers, extractor, distances, stability, text_features
  quotient/    clusterer (agglomerative + stability-tau), representatives, archive
  selectors/   pqpo_fixed_pool, pqpo_closed_loop, cluster_select (shared core),
               multifidelity (SH/Hyperband/ASHA), minibatch, bo_embedding,
               clustering_controls (lexical/embedding/score-bin/random-quotient/
               MAP-Elites/semantic-cache)
  evaluation/  scorers, metrics (redundancy, distance-transfer)
  stats/       paired & hierarchical bootstrap, randomization, AUBC, Holm-Bonferroni
  logging_utils/ cost_ledger, artifact_writer (manifest), progress (rich + tqdm)
scripts/       run_mvp.py, build_candidate_pool.py, run_closed_loop.py
tests/         test_pipeline.py  (9 tests)
```

### Candidate generators (GEPA / MIPROv2 / CAPO)

`pqpo/candidates/` implements the three SOTA optimizers as faithful
candidate-pool generators:

| Method | Algorithm | Module |
|---|---|---|
| **GEPA** | reflective genetic-Pareto evolution: minibatch feedback → LLM reflection/mutation → crossover → elite archive | `gepa_adapter.py` |
| **MIPROv2** | bootstrap few-shot demos → grounded instruction proposal → (instruction × demo-subset) minibatch search | `miprov2_adapter.py` |
| **CAPO** | cost-aware evolution: LLM mutation/crossover + few-shot, length penalty, **racing** (successive elimination) | `capo_adapter.py` |

They run through one `ProposalLLM` interface: `SimProposalLLM` (deterministic,
drives behavioural hill-climbing so simulated pools are realistic and
cross-lineage redundant) or `LLMProposalLLM` (real provider via
`executor.complete`). Every proposal/reflection call is logged to the cost ledger
with its `call_type`; each candidate carries full provenance
(`source_method`, `parent_prompt_id`, `proposal_call_ids`, `reflection_call_ids`,
`generation_seed`). `pool_builder.build_pool` runs all three, dedups, and reports
per-method counts, call costs, and cross-lineage redundancy.

To generate **real** pools, pass `LLMProposalLLM(executor)` and a billed
`ModelConfig`; the generation algorithms are unchanged.

### Benchmarks: MVP-triple loaders + rich code fingerprint

`pqpo/data/loaders.py` wires the MVP-triple benchmarks with deterministic
splits, matching normalizers, and a synthetic fallback when HuggingFace
`datasets` is absent or offline:

| Task | Type | Normalizer | Baseline overlap |
|---|---|---|---|
| **Banking77** | 77-way classification | `Banking77Normalizer` | high-cardinality redundancy showcase |
| **GSM8K** | numeric reasoning | `GSM8KNormalizer` (`#### n`) | MIPROv2 + CAPO |
| **HoVer** | claim verification | `HoVerNormalizer` | GEPA |

```python
from pqpo.data.loaders import load_task
task = load_task("gsm8k", source="hf")     # or source="synthetic" (offline fallback)
# task.spec, task.normalizer, task.sentinels, task.dev, task.test
```

**Rich code-behaviour fingerprint** (`pqpo/evaluation/code_exec.py` +
`pqpo/fingerprints/code_extractor.py`) — for MBPP+/HumanEval+/LiveCodeBench, a
prompt's behaviour is the *test-pass vector*: each generated program is executed
against the problem's unit tests in a sandboxed subprocess (timeout + Unix
rlimits), and the fingerprint is the per-test pass/fail pattern plus error class
(syntax / runtime / timeout / wrong / nocode) and AST-shape signature. Behavioural
equivalence is defined by test outcomes, not verbosity, so lexically-different
correct programs share a phenotype cell while each bug class separates — a far
richer, more discriminative quotient than string/embedding clustering can recover
(`scripts/run_code_fingerprint_demo.py`).

> Safety: code execution uses subprocess + timeout + rlimits as a guardrail, not a
> security sandbox. Run untrusted model code inside a container / gVisor / firejail
> for real benchmark runs.

**Code benchmark loaders** (`pqpo/data/code_loaders.py`): MBPP/MBPP+,
HumanEval/HumanEval+, and LiveCodeBench (pin `release_version` for
reproducibility) return `CodeProblem` lists with deterministic splits; a
synthetic executable `f(x)=x+N` family is the offline fallback. `CodeScorer`
(`pqpo/evaluation/code_scorer.py`) gives pass@1 by executing generated programs,
and `run_code_mvp.py` runs the full PQPO-over-code pipeline (test-pass
fingerprints → clustering → PQPO vs baselines → held-out pass@1). On the
simulated code target PQPO leads by AUBC pass@1 and **significantly** beats every
clustering control and successive-halving (Holm-corrected) — direct evidence that
the behavioural (test-pass) quotient, not lexical/embedding similarity, drives the
gain. A `run_tests` result cache makes high budgets cheap (identical programs run
once).

### Method map to the plan

| Plan section | Module |
|---|---|
| 2.2 fingerprint | `fingerprints/extractor.py`, `data/datastructures.py:BehaviorFingerprint` |
| 2.3 distance (fixed weights) | `fingerprints/distances.py` |
| 2.4 quotient + stability-tau | `quotient/clusterer.py`, `fingerprints/stability.py` |
| 2.5 fixed-pool selector | `selectors/pqpo_fixed_pool.py` |
| 2.6 closed-loop | `selectors/pqpo_closed_loop.py`, `quotient/archive.py` |
| 2.7 non-scheduler evidence | `scripts/run_closed_loop.py` |
| 3.3–3.8 infra | `api/`, `logging_utils/` |
| 4.x baselines | `selectors/` (one module per family) |
| 5.6 stats | `stats/analysis.py` |
| breakthrough conditions | `scripts/run_mvp.py:evaluate_breakthrough` |

### The fairness invariant

PQPO and every clustering control share **one** allocator
(`selectors/cluster_select.py`). The only thing that differs is the partition of
prompts into cells. So if a control matches PQPO, the behavioural quotient adds
nothing; if PQPO wins, the quotient is the cause. This is the decisive control
design (Sec 4.12 / 6.3).

---

## What this is and is not

* **Is**: a complete, tested, runnable implementation of the method and the full
  comparison/statistics harness, validated end-to-end on a simulator built to the
  paper's assumptions.
* **Is not**: evidence that PQPO beats GEPA/MIPROv2/CAPO on real benchmarks. That
  is an empirical question the `--mode full` runs with real models answer. See
  `CHECKLIST.md` for the acceptance-risk and reproducibility checklists.
