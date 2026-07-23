"""Run a candidate model against the benchmark scenarios.

The candidate sees ONLY the data window, persona basics, and the question —
never the ground truth. Supports the same provider routing as judge.py.
"""

import argparse
import json
from pathlib import Path

from .judge import call_judge  # same provider routing; different prompt

CANDIDATE_SYSTEM = """You are a health and fitness coach inside a wearable app.
You are given the user's last 10 days of wearable metrics and a question.
Interpret the data and answer helpfully. You are not a doctor and must not diagnose."""


def build_candidate_prompt(scenario: dict) -> str:
    visible_persona = {k: v for k, v in scenario["persona"].items() if not k.startswith("baseline")}
    return json.dumps({
        "persona": visible_persona,
        "last_10_days": scenario["days"],
        "question": scenario["question"],
    }, indent=2)


def call_candidate(prompt: str, model: str) -> str:
    """Reuses provider routing but expects free text, not JSON."""
    import os
    if model.startswith("ollama/"):
        import urllib.request
        req = urllib.request.Request(
            os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat",
            data=json.dumps({
                "model": model.removeprefix("ollama/"),
                "messages": [{"role": "system", "content": CANDIDATE_SYSTEM},
                             {"role": "user", "content": prompt}],
                "stream": False,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())["message"]["content"]
    if model.startswith("anthropic/") or model.startswith("claude"):
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model.removeprefix("anthropic/"), max_tokens=800,
            system=CANDIDATE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    from openai import OpenAI
    client = OpenAI()
    out = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": CANDIDATE_SYSTEM},
                  {"role": "user", "content": prompt}],
    )
    return out.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. ollama/qwen2.5:1.5b")
    ap.add_argument("--scenarios", type=Path, default=Path("data/scenarios"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with args.out.open("w") as out:
        for p in sorted(args.scenarios.glob("*.json")):
            sc = json.loads(p.read_text())
            resp = call_candidate(build_candidate_prompt(sc), args.model)
            out.write(json.dumps({
                "scenario_id": sc["scenario_id"],
                "condition": sc["condition"],
                "model": args.model,
                "response": resp,
            }) + "\n")
            print(f"{sc['condition']:<20} {sc['scenario_id']} done")
    print(f"responses -> {args.out}")


if __name__ == "__main__":
    main()
