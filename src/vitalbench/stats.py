"""Statistics: bootstrap CIs on axis scores, Cohen's kappa between judges,
and judge validation against the human calibration set.

These are the numbers that make the benchmark credible — and the parts
interviewers actually probe.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

AXES = ["accuracy", "safety", "grounding", "actionability", "calibration"]


def bootstrap_ci(values: list[float], n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    boots = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Agreement between two judges on the same items, corrected for chance."""
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    po = float((a == b).mean())
    pe = sum(float((a == l).mean()) * float((b == l).mean()) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def judge_validation(judge_scores: dict[str, int], human_scores: dict[str, int]) -> dict:
    """Compare a judge's overall_pass / axis scores to human calibration labels.
    Keys are scenario_ids present in both."""
    common = sorted(set(judge_scores) & set(human_scores))
    j = [judge_scores[k] for k in common]
    h = [human_scores[k] for k in common]
    return {
        "n": len(common),
        "kappa": cohens_kappa(j, h),
        "exact_agreement": float(np.mean([x == y for x, y in zip(j, h)])),
        "within_one": float(np.mean([abs(x - y) <= 1 for x, y in zip(j, h)])),
    }


def _load_scored(scored_path: Path) -> dict[str, dict]:
    """scenario_id -> {axis: score, ..., overall_pass: bool, condition: str},
    skipping rubric-validation failures."""
    out = {}
    for line in scored_path.read_text().splitlines():
        rec = json.loads(line)
        s = rec.get("scores", {})
        if "error" in s:
            continue
        entry = {ax: s[ax]["score"] for ax in AXES}
        entry["overall_pass"] = s["overall_pass"]
        entry["condition"] = rec.get("condition", "?")
        out[rec["scenario_id"]] = entry
    return out


def report(scored_path: Path):
    scored = _load_scored(scored_path)
    per_axis = defaultdict(list)
    per_condition = defaultdict(lambda: defaultdict(list))
    passes = []
    for entry in scored.values():
        for ax in AXES:
            per_axis[ax].append(entry[ax])
            per_condition[entry["condition"]][ax].append(entry[ax])
        passes.append(1.0 if entry["overall_pass"] else 0.0)

    print(f"\n{'axis':<15}{'mean':>6}{'95% CI':>18}")
    for ax in AXES:
        m, lo, hi = bootstrap_ci(per_axis[ax])
        print(f"{ax:<15}{m:>6.2f}   [{lo:.2f}, {hi:.2f}]")
    m, lo, hi = bootstrap_ci(passes)
    print(f"{'pass rate':<15}{m:>6.2%}   [{lo:.2%}, {hi:.2%}]")

    print("\nper-condition safety (the slice that matters):")
    for cond, axes in sorted(per_condition.items()):
        m, lo, hi = bootstrap_ci(axes["safety"])
        print(f"  {cond:<20}{m:.2f}  [{lo:.2f}, {hi:.2f}]  n={len(axes['safety'])}")


def _agreement_table(title: str, a: dict[str, dict], b: dict[str, dict]):
    print(f"\n{title}")
    print(f"{'axis':<15}{'n':>5}{'kappa':>8}{'exact':>8}{'within1':>9}")
    for ax in AXES + ["overall_pass"]:
        av = {k: int(v[ax]) for k, v in a.items()}
        bv = {k: int(v[ax]) for k, v in b.items()}
        res = judge_validation(av, bv)
        print(f"{ax:<15}{res['n']:>5}{res['kappa']:>8.2f}{res['exact_agreement']:>8.2%}{res['within_one']:>9.2%}")


def calibration_report(scored_path: Path, human_path: Path):
    from .calibration import load_human_labels
    judge = _load_scored(scored_path)
    human = load_human_labels(human_path)
    if not human:
        print(f"no completed labels in {human_path} yet")
        return
    _agreement_table(f"judge ({scored_path.name}) vs human calibration ({len(human)} labeled)", judge, human)


def agreement_report(scored_a: Path, scored_b: Path):
    a = _load_scored(scored_a)
    b = _load_scored(scored_b)
    _agreement_table(f"judge agreement: {scored_a.name} vs {scored_b.name}", a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, required=True, help="scored jsonl from judge.py")
    ap.add_argument("--calibrate", type=Path, help="human calibration labels (from calibration.py); reports judge-vs-human kappa")
    ap.add_argument("--agreement", type=Path, help="second scored jsonl; reports judge-vs-judge kappa against --scored")
    args = ap.parse_args()

    if args.calibrate:
        calibration_report(args.scored, args.calibrate)
    elif args.agreement:
        agreement_report(args.scored, args.agreement)
    else:
        report(args.scored)


if __name__ == "__main__":
    main()
