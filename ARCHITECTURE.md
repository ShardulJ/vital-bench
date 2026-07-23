# ARCHITECTURE.md

System design for VitalBench. Read alongside MEMORY.md, which specifies the
memory subsystem in depth.

## 1. Overview

VitalBench has two tracks that share one evaluation core:

```
TRACK A: Stateless benchmark (v0.1, shipped)
  synth.py ──> scenarios (10-day windows, labeled ground truth)
                  │
  runner.py ──> candidate model responses (blind to ground truth)
                  │
  judge.py  ──> rubric scores (RubricV1, cached, blind to model identity)
                  │
  stats.py  ──> bootstrap CIs, per-condition slices, judge validation (kappa)

TRACK B: Longitudinal coach benchmark (v0.2, this doc)
  lifesim.py ──> simulated user: 90-365 days of wearable data + event log
                  │                (injuries, illnesses, travel, training blocks,
                  │                 conversation history with embedded facts)
  coach.py   ──> memory-augmented coach agent
                  │     ┌──────────────────────────────┐
                  │     │  MEMORY SUBSYSTEM (MEMORY.md) │
                  │     │  write policy · store · read  │
                  │     │  policy · context assembler   │
                  │     └──────────────────────────────┘
                  │
  probes     ──> memory-dependent questions at day N whose correct answer
                  │  requires information from day N-k (k up to months)
                  ▼
  same judge + stats core, extended with memory-specific rubric axes
```

The point of Track B: every memory strategy in MEMORY.md is a plug-in behind
one interface, so the benchmark produces an *ablation table* — which memory
architecture, at which token budget, keeps a small model accurate and safe
over long horizons. That table is the headline result.

## 2. Module responsibilities

| Module | Owns | Never does |
|---|---|---|
| `synth.py` / `lifesim.py` | Data generation, ground truth, event injection | Any model calls |
| `runner.py` / `coach.py` | Candidate orchestration, memory subsystem | Seeing ground truth |
| `memory/` | Write policy, stores, read policy, context assembly | Model-specific prompt hacks |
| `judge.py` | Scoring against rubric, caching, blindness | Generating candidate answers |
| `stats.py` | CIs, kappa, slices, ablation tables | Interpretation prose |

Hard rule preserved from v0.1: **the candidate path and the ground-truth path
never touch.** In Track B this extends to memory: the memory store is built
only from what the coach observed (data streams + conversations), never from
the event log's hidden labels.

## 3. The coach loop (Track B)

Each simulated day advances in three phases:

1. **Ingest.** The day's wearable summary arrives. The memory write policy
   decides what to persist (see MEMORY.md §3).
2. **Interact (sometimes).** On scripted days the simulated user sends a
   message (a question, a fact disclosure like "I tweaked my knee bouldering",
   or a probe). The coach must respond; the write policy also runs on the
   conversation turn.
3. **Probe (evaluation days).** Probes are questions whose gold answer depends
   on earlier information: "Given my history this month, should I race
   Saturday?" is only answerable correctly if the coach retained the illness
   from day 12 and the disclosed knee injury from day 30.

Probe families (each targets a distinct memory failure mode):
- **Recall probes** — was a disclosed fact retained? (knee injury, medication,
  new job stress)
- **Trend probes** — does the coach compare against the *user's own* baseline
  from weeks ago rather than population norms?
- **Contradiction probes** — user says something inconsistent with history
  ("I've been sleeping great" after 10 logged bad nights); safe coaches
  gently ground in data, unsafe ones agree.
- **Staleness probes** — the injury healed on day 45; is the coach still
  restricting training on day 80? Memory must support *decay/invalidation*,
  not just accumulation.
- **Interference probes** — two similar events (two illnesses, two travel
  blocks); does retrieval confuse them?

## 4. Interfaces

```python
class MemoryStrategy(Protocol):
    def write(self, event: Observation) -> None: ...
    def read(self, query: str, today: Date, budget_tokens: int) -> AssembledContext: ...

@dataclass
class AssembledContext:
    system_suffix: str          # profile / standing facts
    history_block: str          # serialized longitudinal data
    retrieved_block: str        # episodic snippets, if any
    token_count: int
    provenance: list[str]       # which store entries were used (for grounding checks)
```

`provenance` matters: the judge's grounding axis can verify that facts cited
by the coach exist in the store, catching memory-hallucination separately from
data-hallucination.

## 5. Evaluation extensions

RubricV2 adds two axes to the five in RubricV1:
- **retention** — did the response use required past information (probe-specific)?
- **currency** — did it avoid using invalidated/stale information?

All Track A machinery (blind judging, caching, kappa validation, bootstrap
CIs) applies unchanged. The ablation report groups results by
(memory_strategy × token_budget × model), with per-probe-family slices.

## 6. Constraints honored throughout

- Runs on an 8GB MacBook Air: candidates are <=3B local models or API calls;
  embedding retrieval uses a small sentence-transformers model on CPU;
  stores are SQLite + JSON, no vector DB dependency.
- Judge cost stays low: probes are sparse (5-10 per simulated user), caching
  is content-hash based, and the ablation grid is sampled, not exhaustive.
- Determinism: lifesim is seeded; a (seed, strategy, budget, model) tuple
  fully reproduces a run.

## 7. Non-goals

- No real user data, ever (privacy and no ground truth).
- No claim of clinical validity; docs/physiology.md documents simplifications.
- Not a memory *product* — a benchmark of memory *strategies*. Keep each
  strategy implementation minimal and readable over feature-rich.
