# Fine-tuning track: closing the gap on a MacBook Air

Goal: show that a small model fine-tuned on eval-derived data closes a measurable
fraction of the gap to a frontier model on VitalBench — trained entirely on an
8GB Apple Silicon MacBook Air with MLX.

## Memory budget (8GB unified)

| Component | Approx |
|---|---|
| Qwen2.5-1.5B-Instruct, 4-bit | ~1.0 GB |
| LoRA adapters + optimizer state | ~0.3 GB |
| Activations at batch 1, seq 1024 | ~1.5 GB |
| macOS + headroom | ~4 GB |

1.5B at 4-bit with batch size 1 and gradient checkpointing fits. Phi-3-mini
(3.8B) is possible but tight — close everything else. Start with Qwen2.5-1.5B.

## Setup

```bash
pip install mlx-lm
python -m mlx_lm.generate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --prompt "hello"  # smoke test
```

## Step 1 — Build the SFT dataset from the benchmark itself

Generate 300+ scenarios, get responses from a strong teacher (Claude/GPT-4 class,
~$3-5 with caching), keep only responses the judge scores >=4 on safety AND
accuracy, and format as chat pairs:

```bash
python -m vitalbench.synth --n 300 --out data/train_scenarios/
python -m vitalbench.runner --model claude-sonnet-4-6 --scenarios data/train_scenarios/ --out results/teacher.jsonl
python -m vitalbench.judge --responses results/teacher.jsonl --judge claude-haiku-4-5-20251001 --out results/teacher_scored.jsonl
python finetune/make_sft_data.py --scored results/teacher_scored.jsonl --min-safety 4 --out finetune/data/
```

This is the loop worth talking about in interviews: the benchmark generates its
own filtered training data (rejection sampling against the rubric).

## Step 2 — LoRA SFT with MLX

```bash
python -m mlx_lm.lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train --data finetune/data \
  --batch-size 1 --grad-checkpoint \
  --num-layers 8 --iters 600 \
  --adapter-path finetune/adapters_sft
```

Expect roughly 1-2 hours on an M1/M2 Air. If memory pressure hits, drop
--num-layers to 4 (fewer layers get adapters).

## Step 3 — DPO pairs from judge disagreements

For each training scenario you now have multiple responses (teacher, base
small model, SFT small model) with judge scores. Build preference pairs:
chosen = highest-scoring, rejected = a response failing safety or grounding.
That is a DPO dataset built from real eval failures rather than synthetic
preferences — the exact methodology story that distinguishes you.

```bash
python finetune/make_dpo_pairs.py --scored results/ --out finetune/dpo_data/
python -m mlx_lm.lora --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train --data finetune/dpo_data --training-mode dpo \
  --batch-size 1 --grad-checkpoint --adapter-path finetune/adapters_dpo
```

(If your mlx-lm version lacks DPO mode, pin a recent version or fall back to
a second SFT round on chosen-only data and say so honestly in the writeup.)

## Step 4 — Evaluate base vs SFT vs DPO on the held-out benchmark

```bash
python -m mlx_lm.fuse --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --adapter-path finetune/adapters_dpo --save-path finetune/qwen-vital
# serve locally via mlx_lm.server (OpenAI-compatible), then:
python -m vitalbench.runner --model qwen-vital --scenarios data/scenarios/ --out results/qwen_dpo.jsonl
python -m vitalbench.judge --responses results/qwen_dpo.jsonl --judge claude-haiku-4-5-20251001 --out results/qwen_dpo_scored.jsonl
python -m vitalbench.stats --scored results/qwen_dpo_scored.jsonl
```

The headline chart for the README and your LinkedIn post: safety and accuracy,
base -> SFT -> DPO, with bootstrap CIs, next to the frontier-model ceiling.

## Honest-limitations section (write this — it impresses more than the results)

- Judge and teacher share a model family: measure kappa against your human
  calibration set and report it.
- Synthetic physiology is a simplification: link docs/physiology.md.
- Held-out scenarios use unseen seeds but the same generator: note the
  distribution-shift caveat and propose real-data validation as future work.
