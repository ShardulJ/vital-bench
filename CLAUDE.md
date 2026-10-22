# CLAUDE.md — vitalbench

## Project
Open benchmark for evaluating LLM safety on wearable physiological data.
Read README.md, ARCHITECTURE.md and MEMORY.md before any task.

## Commit conventions
- One logical change per commit. Short imperative subject lines.
- Always run the smoke test before committing:
  PYTHONPATH=src python -m vitalbench.synth --n 2 --out /tmp/smoke/

## Code style
- Python 3.10+, type hints, no classes where functions suffice.
- Keep modules under ~200 lines. Prefer stdlib over new dependencies.
- No dead code, no TODO comments without an issue reference.

## Working style
- Propose a plan before writing code. Wait for my approval.
- When I ask "why", defend the design choice or change it.
- I write docs/physiology.md, rubric definitions, and README design
  sections myself. You may review them but not rewrite them.
- Never push to GitHub unless I explicitly say "push" or "push this."
