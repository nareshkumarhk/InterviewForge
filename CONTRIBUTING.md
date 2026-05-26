# Contributing to InterviewForge

Thanks for contributing 👋

This repository contains source code and **evaluation tooling** for interview analysis. To keep history clean and reviews fast, please follow the rules below.

## Development setup

1. Work from the repository root.
2. For the Python project, use:
   - `phase0/interview-review-agent/requirements.txt`
   - `.env.example` as your template for local `.env`
3. Never commit secrets.

## What should be committed

- Source code under `phase0/interview-review-agent/src/`
- Scripts under `phase0/interview-review-agent/scripts/`
- Prompt files and changelog under `phase0/interview-review-agent/prompts/`
- Specs and documentation updates
- Test fixtures in `phase0/interview-review-agent/tests/transcripts/`

## What should NOT be committed

Generated artifacts are intentionally ignored. Do **not** stage these:

- `phase0/interview-review-agent/tests/results/partial/*`
- `phase0/interview-review-agent/tests/results/v*/` outputs (`.json`, `.md`, etc.)
- Local environment files (`.env`, `.env.*` except `.env.example`)
- Editor and OS artifacts (`.DS_Store`, `.vscode/`, `.idea/`)

If you need example output structure, keep only placeholder `.gitkeep` files where present.

## Pull request checklist

Before opening a PR:

- [ ] Run relevant scripts/tests for your change
- [ ] Confirm no generated output files were staged
- [ ] Confirm no secrets are included
- [ ] Update docs when behavior or workflow changes

## Commit style

Prefer clear, scoped commit messages, e.g.:

- `Improve evaluation decision logic for prompted answers`
- `Update prompt v1.4 scoring rubric`
- `Add transcript fixture for lead .NET scenario`

Small, focused PRs are easier to review and safer to merge.
