# MEMORY.md

Design of the memory subsystem for the longitudinal coach track. The research
question the ablation answers: **which memory architecture, at which token
budget, keeps a small model accurate and safe over months of physiological
data?**

Every strategy implements the same `MemoryStrategy` interface (ARCHITECTURE.md
§4) so the benchmark compares them under identical conditions.

## 1. The problem, concretely

A year of daily wearable summaries is ~150-400K tokens serialized naively -
beyond most small-model context windows, and even where it fits, models
degrade on long contexts (lost-in-the-middle effects) and cost scales
linearly. Conversations add facts that never appear in the data stream
("I tweaked my knee", "I started a new medication"). A coach therefore needs:
a **write policy** (what to remember), a **store** (how to hold it), a
**read policy** (what to recall for this query), and a **context assembler**
(how to fit it in budget). Each strategy below makes different choices.

## 2. Strategies under test

### M0 - Full context (upper-bound baseline)
Everything, serialized, most recent last. Only feasible for short horizons or
long-context API models. Establishes the ceiling and the cost curve.

### M1 - Sliding window + rolling summary
Keep last N days raw; everything older is folded into a running summary that
is *rewritten* (not appended) each week by the model itself.
- Write: append day; weekly, compress oldest week into the summary.
- Failure mode probed: rewriting loses specifics (exact dates, the disclosed
  injury), and errors compound - summary-of-summary drift.

### M2 - Structured profile memory (schema-first)
An explicit typed store the model writes to via extraction after every
observation/conversation:

```python
class UserProfile(BaseModel):
    baselines: dict[str, RollingBaseline]        # per-metric, updated daily, cheap math not LLM
    conditions: list[Fact]                        # injuries, illnesses, meds
    context: list[Fact]                           # job stress, travel, goals
    preferences: list[Fact]                       # training style, tone

class Fact(BaseModel):
    text: str
    source: str            # "conversation:day30" | "inferred:day12"
    created: Date
    status: Literal["active", "resolved", "superseded"]
    expiry_hint: Date | None
```

- Write: LLM extraction into the schema; numeric baselines updated by plain
  code (no LLM - cheaper and exact).
- Read: whole profile always ships (it's small); that's the point.
- Key design element: **facts have lifecycle status.** Resolution ("knee feels
  fine now") flips status instead of deleting - supports staleness probes and
  auditability.
- Failure mode probed: extraction misses implicit facts; schema can't hold
  what it didn't anticipate.

### M3 - Episodic retrieval (RAG over the past)
Every day-summary and conversation turn is chunked and embedded (small CPU
sentence-transformers model, SQLite + numpy - no vector DB). At read time,
retrieve top-k by a blended score:

```
score = α·cosine(query, chunk) + β·recency_decay(age) + γ·salience(chunk)
```

salience is set at write time (abnormal days score higher - deviation from
baseline in σ units), so a routine Tuesday ranks below the day HRV cratered.
- Failure mode probed: interference (retrieves the wrong illness), and the
  trend-blindness of chunk retrieval - episodic memory sees trees, not forests.

### M4 - Hierarchical temporal memory (summaries at multiple resolutions)
Days roll into week summaries, weeks into month summaries; all levels are
retained. Reads assemble coarse-to-fine: months for background, weeks for the
recent past, raw days for the immediate window. This mirrors how coaches
actually reason ("a rough month, a better fortnight, a bad yesterday").
- Failure mode probed: whether multi-resolution beats M1's single rolling
  summary at equal token budget - the cleanest ablation pair in the study.

### M5 - Agentic memory (model-managed via tools)
The coach gets `remember(fact)`, `recall(query)`, `update_status(fact_id)`
tools and decides for itself what to store and fetch. The most 2026-shaped
design and the least controllable; included because everyone claims it works
and few measure it - especially for <=3B models, where tool-call reliability
is itself in question.

## 3. Context assembly (shared machinery)

Assembly is where most quality is won or lost, independent of store choice:

1. **Token budgeting by tiers.** Fixed budget split: standing profile (~15%),
   longitudinal history (~50%), retrieved episodes (~25%), headroom (~10%).
   Tiers degrade gracefully: overflow drops lowest-salience items first,
   never truncates mid-item.
2. **Time-series serialization.** Raw JSON per day is ~40 tokens/metric-day of
   waste. Instead: recent days as a compact aligned table; older periods as
   computed statistics (mean, trend slope, notable deviations) - *code*
   computes these, the model never re-derives arithmetic it will get wrong.
   Downsampling schedule: last 7 days raw, last 8 weeks weekly, older monthly.
3. **Placement against lost-in-the-middle.** Standing facts and safety-critical
   items (active injury, medication) go at the *end* of the assembled context,
   adjacent to the question; bulk history sits earlier. Middle positions get
   the least-critical material.
4. **Provenance threading.** Every assembled block carries store-entry IDs so
   the grounding judge can verify cited facts exist (memory-hallucination
   detection, distinct from data-hallucination).
5. **Query-conditioned assembly.** The read policy sees the user's question
   first; a race-planning question weights training-load history, a "should I
   be worried" question weights the illness/anomaly record. Implemented as a
   cheap keyword/intent gate, not another LLM call.

## 4. What the ablation reports

For each (strategy × budget ∈ {2K, 6K, 16K} × model):
- RubricV2 axes with bootstrap CIs, sliced by probe family
  (recall / trend / contradiction / staleness / interference)
- tokens consumed per interaction (cost curve)
- write-time overhead (extraction/summarization calls per day)

Hypotheses stated before running (pre-registration, in this file's git
history - models can be wrong in public, that's the credibility play):
- H1: M2 (structured) dominates on staleness and contradiction probes at all
  budgets, because lifecycle status is explicit.
- H2: M3 (episodic) wins recall but loses trend probes to M4 at equal budget.
- H3: M5 (agentic) underperforms its reputation for <=3B models due to
  tool-call reliability, and the gap closes with model scale.
- H4: assembly placement (§3.3) alone moves safety scores by a measurable
  margin holding the store constant - context engineering matters as much as
  memory architecture.

## 5. Build order

1. `memory/base.py` (interface, AssembledContext) + M0 + M1 - one afternoon.
2. lifesim event log + the five probe families + RubricV2 - the real work; do
   this before adding more strategies, because probes define what "better"
   means.
3. M2, then M3 (reuse sentence-transformers from apply_job), then M4.
4. M5 last, only if tool-calling in the chosen small models is stable enough
   to measure rather than babysit.
5. Ablation runs are overnight jobs on the Mac; the grid is sampled (not every
   cell) - report which cells ran.

## 6. Honest limitations (keep in the writeup)

- Simulated users disclose facts in clean sentences; real users are messier.
- Probe families are hand-designed; a real deployment has unknown-unknowns.
- One judge family scores all strategies; kappa vs the human calibration set
  is reported per probe family, and probe wording avoids strategy-identifying
  cues so judging stays blind.
