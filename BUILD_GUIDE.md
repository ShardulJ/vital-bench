# VitalBench build & launch manual

Four weeks, part-time, alongside the job search. Total cash cost: $5-15.

## Week 1 — Core loop working end to end
- Get synth -> runner -> judge -> stats running with 20 scenarios and one free
  local candidate (ollama/qwen2.5:1.5b) and one cheap API judge.
- Write docs/physiology.md: one paragraph per condition justifying the
  parameter ranges, citing published HRV/overtraining/illness-detection
  literature. This file is what separates "toy" from "benchmark" — hiring
  managers in healthtech will read it first.
- Cost this week: ~$1 of judge calls.

## Week 2 — Make the judge trustworthy
- Hand-label 30 scenarios yourself (you are the human calibration set v1;
  say so honestly). Score them with 2-3 different judges.
- Report Cohen's kappa judge-vs-you and judge-vs-judge. Tune the rubric
  wording until kappa > 0.6, documenting what changed and why — keep a
  CHANGELOG in the rubric docstring.
- Add the position-bias check: run 20 pairwise comparisons twice with order
  swapped; report the flip rate.
- Cost: ~$3.

## Week 3 — The MacBook fine-tuning track
- Follow finetune/README.md: teacher generation, rejection-sampled SFT set,
  MLX LoRA, DPO pairs from judge failures, before/after evaluation.
- Produce the headline chart: base vs SFT vs DPO vs frontier ceiling, with CIs.
- Cost: ~$5 teacher + judge calls. Training is free (your Mac, overnight).

## Week 4 — Launch
- Seed the README leaderboard with 5-6 models: two local small models, your
  fine-tuned variants, one or two frontier models via API free tiers/cheap calls.
- Write the launch post (see below), pin repo + post to LinkedIn Featured.
- File one issue on a related OSS project (mlx-lm, an eval framework) linking
  your benchmark — small, legitimate visibility loops.

## The launch post (LinkedIn + a longer version on GitHub/blog)

Structure that works:
1. The gap: "Wearable LLM coaches are shipping. Nobody has published how to
   evaluate whether their advice is safe."
2. What you built, in three bullets.
3. One surprising finding (there will be one — e.g., small models fabricate
   data values far more than they give unsafe advice, or every model
   over-alarms on normal variation).
4. The chart.
5. "Repo is open, PRs welcome, here's what I'd build next."

Tag nothing, pitch nothing, ask for nothing. Recruiters and hiring managers
find technical posts with real findings; they ignore posts that smell like
job-seeking.

## How this maps to interview questions you were weak on

| Interview probe | Your concrete answer now |
|---|---|
| "How do you evaluate an LLM system?" | Rubric schemas, blind judging, kappa validation, bootstrap CIs, per-slice reporting — all in a repo you can screen-share |
| "Tell me about fine-tuning you've done" | ChiEAC (production) + VitalBench (public, reproducible, with ablation: base vs SFT vs DPO) |
| "How do you build preference data?" | DPO pairs mined from judge-scored failures; rejection sampling for SFT |
| "How do you know your judge is right?" | Human calibration set, kappa, position-bias flip rate, honest limitations doc |
| "Resource constraints?" | Quantization + LoRA memory budgeting to fit 8GB unified memory, documented |

## Resume line (once results exist)

VitalBench  |  github.com/ShardulJ/vitalbench
- Open benchmark for LLM safety on wearable physiological data: synthetic
  scenario generator with labeled ground truth, validated LLM-as-judge harness
  (Cohen's kappa 0.7 vs human labels), bootstrap CIs, per-condition safety slicing.
- Fine-tuned Qwen2.5-1.5B with QLoRA SFT and DPO on eval-derived preference
  pairs using MLX on an 8GB MacBook Air, improving safety pass rate from X% to Y%.

(Replace X/Y with real numbers. Never publish the line before the numbers exist.)
