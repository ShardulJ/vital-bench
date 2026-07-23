# VitalBench

**An open evaluation benchmark for LLMs that interpret wearable physiological data.**

Wearable companies are shipping LLM "coaches" that read your HRV, sleep, and strain data and tell you what to do. Almost none of them can answer a basic question: *how do you know the model's interpretation is safe and correct?*

VitalBench answers it with three components:

1. **Synthetic physiological scenarios with known ground truth.** A generator produces realistic multi-day wearable data (HRV, resting heart rate, sleep stages, strain/recovery) with injected, labeled conditions: overtraining onset, illness prodrome, alcohol impact, circadian disruption, normal variation. Because the condition is injected, the correct interpretation is known — no expensive human labeling required.

2. **Safety-first rubrics with structured scoring.** Each model response is scored on five axes: physiological accuracy, safety (does it flag what needs a doctor, does it avoid diagnosing), actionability, grounding (does it cite the actual data or hallucinate values), and calibration (does its confidence match the evidence). Rubrics are Pydantic schemas — machine-checkable, versioned, reproducible.

3. **A validated LLM-as-judge harness.** Judges are not trusted blindly: the harness measures inter-judge agreement (Cohen's kappa), validates judges against a human-labeled calibration set, controls for position bias in pairwise comparisons, and reports bootstrap confidence intervals on every score.

Plus: a fine-tuning track showing that a 1.5B model fine-tuned with QLoRA **on a MacBook Air (MLX)** closes a measurable fraction of the gap to frontier models on this benchmark — with the full DPO dataset built from eval failures.

## Why this exists

Evaluation is the bottleneck for health LLMs. General benchmarks (MMLU, HealthBench-style medical QA) don't test the core wearable task: *longitudinal reasoning over an individual's physiological time series*. A model can ace medical trivia and still tell an overtrained athlete to push harder because it misread an HRV trend. VitalBench tests exactly that failure mode.

## Quickstart

```bash
pip install -e .

# 1. Generate 50 scenarios across all condition types
python -m vitalbench.synth --n 50 --out data/scenarios/

# 2. Run a model against them (any OpenAI-compatible endpoint, Anthropic, or Ollama)
python -m vitalbench.runner --model ollama/qwen2.5:1.5b --scenarios data/scenarios/ --out results/qwen_base.jsonl

# 3. Judge the responses (cached, ~$0.02 per scenario with a Haiku-class judge)
python -m vitalbench.judge --responses results/qwen_base.jsonl --judge claude-haiku --out results/qwen_base_scored.jsonl

# 4. Report with bootstrap CIs
python -m vitalbench.stats --scored results/qwen_base_scored.jsonl
```

## Leaderboard (seed results)

| Model | Accuracy | Safety | Grounding | Overall (95% CI) |
|---|---|---|---|---|
| *populate with your runs* | | | | |

## Design decisions

- **Synthetic over real data.** Real wearable data has privacy problems and no ground truth. Synthetic generation with physiologically plausible parameters (documented in `docs/physiology.md`) trades some realism for perfect labels and infinite scale. The calibration set of 30 human-reviewed items anchors the judge to human judgment.
- **Judges are measured, not assumed.** Every reported score ships with the judge's kappa against the calibration set. If the judge can't reproduce human labels, its scores are flagged.
- **Small-model track is first-class.** The interesting question for on-device health coaching isn't "is GPT-4 good" — it's "how close can a model that runs on a watch or phone get." All baselines include sub-3B models.

## Repo structure

```
vitalbench/
├── src/vitalbench/
│   ├── synth.py        # synthetic wearable data generator with condition injection
│   ├── scenarios.py    # condition definitions + ground-truth interpretations
│   ├── rubric.py       # Pydantic scoring schemas
│   ├── judge.py        # LLM-as-judge harness with caching and bias controls
│   ├── runner.py       # runs candidate models against scenarios
│   └── stats.py        # bootstrap CIs, Cohen's kappa, judge validation
├── finetune/           # MLX QLoRA fine-tuning on Apple Silicon (see finetune/README.md)
├── data/scenarios/     # generated benchmark items
├── docs/physiology.md  # parameter justification with citations
└── results/
```

## License

MIT
