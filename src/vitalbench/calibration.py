"""Human calibration set: template generation and loading.

The human calibration set is you, hand-scoring a fixed batch of
(scenario, candidate response) pairs with the same rubric axes the judge
uses. Judges are validated against these labels via stats.judge_validation.

Workflow:
  1. python -m vitalbench.calibration --responses results/qwen1.5b_candidate.jsonl \
         --out data/calibration/human_labels.json --n 30
  2. Open the file, fill in each null score (1-5) and overall_pass (true/false)
     by reading scenario + response yourself.
  3. Score the same responses with 2-3 judges (judge.py) and compare via
     stats.py's --calibrate flag.
"""

import argparse
import json
from pathlib import Path

from .rubric import RubricV1

AXES = list(RubricV1.model_fields)
AXES = [a for a in AXES if a != "overall_pass"]


def make_template(responses_path: Path, scenarios_dir: Path, n: int) -> list[dict]:
    scen = {json.loads(p.read_text())["scenario_id"]: json.loads(p.read_text())
            for p in scenarios_dir.glob("*.json")}

    records = [json.loads(line) for line in responses_path.read_text().splitlines()][:n]
    template = []
    for rec in records:
        sc = scen[rec["scenario_id"]]
        template.append({
            "scenario_id": rec["scenario_id"],
            "condition": rec["condition"],
            "question": sc["question"],
            "response": rec["response"],
            "ground_truth": sc["ground_truth"],
            "human_scores": {axis: None for axis in AXES},
            "overall_pass": None,
            "notes": "",
        })
    return template


def load_human_labels(path: Path) -> dict[str, dict]:
    """Returns scenario_id -> {axis: score, ..., overall_pass: bool}, skipping
    entries that haven't been labeled yet (any null field)."""
    entries = json.loads(path.read_text())
    labels = {}
    for e in entries:
        if e["overall_pass"] is None or any(v is None for v in e["human_scores"].values()):
            continue
        labels[e["scenario_id"]] = {**e["human_scores"], "overall_pass": e["overall_pass"]}
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, required=True, help="jsonl from runner.py")
    ap.add_argument("--scenarios", type=Path, default=Path("data/scenarios"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; refusing to overwrite existing labels")

    template = make_template(args.responses, args.scenarios, args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(template, indent=2))
    print(f"wrote {len(template)}-item calibration template -> {args.out}")
    print("fill in human_scores and overall_pass for each entry before running stats --calibrate")


if __name__ == "__main__":
    main()
