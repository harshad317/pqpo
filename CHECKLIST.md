# PQPO — Conference Checklists (NeurIPS / ICLR)

**Status note (June 2026):** The NeurIPS 2027 and ICLR 2027 calls are not yet
published. The NeurIPS Paper Checklist has been stable across 2023–2026 (the
mandatory ~15-question form below), and ICLR's requirements (reproducibility
statement, ethics statement, Code of Ethics conformance) have likewise been
stable. This document fills the **current** NeurIPS checklist for PQPO and lists
the ICLR-specific items; re-confirm against the 2027 author kits when released.

Legend for "Repo status": ✅ satisfied by this codebase · 🔬 requires the
`--mode full` real-model runs · ✍️ a writing task for the paper.

---

## A. NeurIPS Paper Checklist (answer Yes/No/NA + 1–2 sentence justification)

**1. Claims** — *Do abstract/intro claims match contributions and scope?*
**Answer: Yes.** ✍️ The three contributions (behavioral quotient as optimization
state; API-efficient fixed-pool selection; closed-loop quotient-guided
optimization) are exactly what the method and experiments demonstrate. Do **not**
claim first-ever phenotypes/QD/semantic-caching (Sec 1 "must not claim"); claim
only "target-model-induced behavioral equivalence classes as first-class
optimization states."

**2. Limitations** — *Are limitations discussed?*
**Answer: Yes.** ✍️ Required limitations to state: sentinel probes add cost (must
be counted); quotient can overfit the sentinel panel; API nondeterminism can
destabilize cells; candidate pools may be artificial; results are model- and
task-family specific. See `pqpo/.../failure modes` mapped from Sec 7.

**3. Theory, assumptions, proofs** — **Answer: NA (mostly).** ✍️ The quotient is a
formal construction (Sec 2.1), not a theorem. If you include the cluster-stability
or distance-transfer claims as propositions, give assumptions + proof in the
appendix; otherwise mark NA and present them as empirical.

**4. Reproducibility** — **Answer: Yes.** ✅🔬 Full method, baselines, stats, and
artifact schema are released. Disclose: candidate-pool construction, sentinel
panel, splits, fingerprint schema + fixed distance weights, tau-selection
procedure, selector hyperparameters (β=0.5, λ=0.1, ρ=0.1, γ=0.05 — untuned),
seeds. Deterministic request hashing + cache make runs replayable.

**5. Open access to data & code** — **Answer: Yes.** ✅🔬 This repo is the code.
For submission: release candidate pools, sentinel outputs, fingerprints, quotient
clusters, selector runs, cost ledgers, and final-test outputs (Sec 10.1), with an
anonymized repo for double-blind review.

**6. Experimental settings/details** — **Answer: Yes.** ✅✍️ Specify per task:
dataset version, 240 prompts/task, 12 sentinels, 96 selector-dev, 300–500 test,
budgets {0,240,480,960}, decoding (temperature 0, max tokens), provider + model
version, pricing snapshot. The manifest writer records provenance automatically.

**7. Statistical significance** — **Answer: Yes.** ✅🔬 Report paired bootstrap
95% CIs, hierarchical bootstrap across (task, seed, example), paired randomization
win-rate, AUBC CIs, cluster-stability ARI/NMI CIs, distance-transfer correlation
CIs, with **Holm–Bonferroni** across all PQPO-vs-baseline comparisons
(`pqpo/stats/analysis.py`). No single-seed wins (≥5 seeds).

**8. Compute resources** — **Answer: Yes.** 🔬✍️ Report total target-model calls,
tokens, dollar cost (split by call type), wall-clock, and any proposal/reflection
LLM compute. The CostLedger produces these tables; include the precompute-vs-
algorithm-visible distinction (Sec 5.5).

**9. Code of Ethics** — **Answer: Yes.** ✍️ No human subjects, no PII; uses public
benchmarks under their licenses. Confirm conformance after reading the current
NeurIPS Code of Ethics.

**10. Broader impacts** — **Answer: Yes.** ✍️ Positive: cheaper prompt
optimization reduces API cost/energy. Negative: more efficient optimization could
lower the cost of optimizing harmful prompts; discuss concretely, not generically.

**11. Safeguards** — **Answer: NA / Yes.** ✍️ No high-risk model or scraped data
released. If you release candidate-generation prompts, note any misuse surface.

**12. Licenses for existing assets** — **Answer: Yes.** ✍️ Cite and honor licenses
for TREC, GSM8K-style, extraction datasets, and any baseline code (GEPA/DSPy-
MIPROv2/CAPO). Record versions.

**13. New assets** — **Answer: Yes.** ✅✍️ New assets = candidate pools,
fingerprints, quotient clusters, this code. Document schemas (Sec 10.3) and
release with a license.

**14. Crowdsourcing / human subjects** — **Answer: NA.** No human-subject data.

**15. IRB approvals** — **Answer: NA.** No human-subjects research.

**16. LLM usage declaration** — **Answer: Yes.** ✍️ Declare LLMs used as the
target model, proposal/reflection models, and (if any) writing assistance, with
versions and dates.

---

## B. ICLR-specific items

- **Reproducibility Statement** (required, uncounted): one paragraph pointing to
  the code, data, splits, and appendix details. ✅🔬 — covered by Sections A.4–A.6.
- **Ethics Statement** (optional but expected for dual-use): ✍️ — reuse A.9–A.11.
- **Code of Ethics conformance**: ✍️ confirm against the ICLR code.
- **Anonymization**: ✅ remove author/provider account identifiers from artifacts
  and cost ledgers before submitting the supplementary.
- **No appendix page limit**: put full baseline-repro details, ablation tables,
  and stability analyses in the appendix.

---

## C. Acceptance-risk checklists (from the implementation plan, Sec 11)

### C.1 Originality
- ✅ Optimization state = behavioral equivalence class (not string/embedding/score-bin/lineage).
- ✍️ Cede prior art on phenotypes, QD/MAP-Elites, semantic caching, multi-fidelity; claim only the quotient-as-state framing.

### C.2 Significance (evidence required) 🔬
- Large behavioral redundancy in **real** optimizer proposal streams.
- Lower held-out cost at equal score; wins across classification, reasoning,
  extraction, and code/tool/RAG; positive across ≥5 seeds; savings survive
  sentinel accounting.

### C.3 Technical depth (must all be present) ✅
formal quotient · task-independent fingerprint · distance metric · stability-based
tau · representative selection · cell-level uncertainty · novelty/exploration
pressure · phenotype-targeted mutation · cost-normalized comparison · mechanistic
transfer analysis. *("cluster prompts → eval one per cluster" alone is not enough.)*

### C.4 Baselines (must include) ✅ implemented; 🔬 real-model repro pending
GEPA · MIPROv2 · CAPO · MO-CAPO · MAP-Elites · Hyperband · ASHA · successive
halving · BO over embeddings · random & stratified minibatch · lexical/embedding/
semantic-cache/score-bin/lineage/random-quotient controls.
*Implemented in `selectors/`; the three SOTA optimizers (GEPA/MIPROv2/CAPO) need
adapter wiring to their real candidate-generation streams for `--mode full`.*

### C.5 Statistics ✅
paired bootstrap CIs · randomization tests · Holm–Bonferroni · seed & task
variance · AUBC CIs · cluster-stability CIs · distance-transfer CIs. No single-seed wins.

### C.6 Reviewer objections & rebuttals (Sec 7) ✍️
- *"Just MAP-Elites"* → behavioral (not structural) descriptors; MAP-Elites
  control with structural descriptors is included and must fail to match PQPO.
- *"Just Hyperband/ASHA + clustering"* → closed-loop targets mutation by phenotype
  state; report target-cell hit rate, new-cell entry rate, proposal-distribution
  shift, and selector-only-vs-targeted at equal cost (`scripts/run_closed_loop.py`).
- *"Sentinels aren't free"* → all sentinel calls counted in the dollar ledger; the
  ≤105%-of-best-baseline total-cost bar is a reported metric.
- *"Lexical/embedding clustering matches PQPO"* → both are run through the **same**
  allocator; they must underperform (C4 in the MVP).

---

## D. Breakthrough / kill gate (run before scaling)

The MVP (`scripts/run_mvp.py`) must pass all six before closed-loop or full
baselines: (1) redundancy ≥30% in ≥2 tasks; (2) PQPO beats best non-behavioral by
+10% AUBC **or** −40% labeled evals; (3) total cost ≤105% of strongest baseline
including sentinels; (4) all quotient controls fail; (5) phenotype distance is the
strongest transfer predictor; (6) cluster stability ARI ≥0.65 (MVP) / ≥0.75
(paper). On the bundled simulator, conditions 1, 2, 4, 5, 6 pass and 3 is a
cost-ledger readout on real runs.

---

*Sources: NeurIPS 2026 Paper Checklist (neurips.cc Main Track Handbook;
checklist.tex author kit). Re-verify against NeurIPS/ICLR 2027 author kits on
release.*
