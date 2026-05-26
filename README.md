# InterviewForge

AI-powered interview evaluation toolkit for technical hiring.

This repository currently contains **Phase 0** of the Interview Review Agent, focused on validating that AI-generated interview evaluations are accurate, evidence-backed, and useful for hiring decisions.

## What this project does

- Analyzes interview transcripts using prompt-driven AI steps
- Produces structured dimension scores and hiring recommendations
- Adds deterministic decision logic and hallucination checks
- Supports prompt version comparisons and accuracy measurement against ground truth

## Repository layout

- `phase0/` — phase specifications and implementation
- `phase0/interview-review-agent/` — executable Python project
	- `src/` — core engine and decision logic
	- `scripts/` — CLI scripts for running evaluations and metrics
	- `prompts/` — versioned prompts (`v1.0` ... `v1.3`)
	- `tests/transcripts/` — transcript fixtures + metadata + ground truth
	- `tests/results/` — generated evaluation outputs (ignored by git)

## Quick start

1. Go to project folder:
	 - `phase0/interview-review-agent/`
2. Create and activate virtual environment
3. Install dependencies from `requirements.txt`
4. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`

## Common workflows

- Run an evaluation: `scripts/run_evaluation.py`
- Measure accuracy: `scripts/measure_accuracy.py`
- Compare prompt versions: `scripts/compare_versions.py`
- Check model/API rate limits: `scripts/check_rate_limits.py`

For detailed setup, CLI flags, and pipeline internals, see:

- `phase0/interview-review-agent/README.md`

## Notes

- Generated evaluation outputs in `tests/results/` are intentionally ignored to keep the repository lean and avoid noisy diffs.
- Keep only canonical fixtures (transcripts + ground truth) under version control.
