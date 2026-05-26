# Prompt Iteration Log

> **This file is no longer maintained.**
> All prompt change history, test results, and version comparisons are now tracked in
> [`prompts/CHANGELOG.md`](../prompts/CHANGELOG.md).

---

## v1.1 (2026-05-17)

### Status: In testing

### Problem observed (v1.0 failure on Manas / .NET Lead)
1. **False floor at 2.0**: Skills where candidate explicitly admitted no experience were scored 2.0. The rubric said "0-1: cannot solve / no understanding" but lacked a clear rule for "I haven't worked with X" — model defaulted to 2.0 as a conservative middle ground.
2. **Behavioral blind spot**: Communication scored only on technical clarity. Behavioral signals (conflict handling, feedback, collaboration) were never extracted from the transcript and never scored — making a lead candidate look average when they may have been weaker or stronger.

### Fixes in v1.1
- Hard rule added: admitted no experience = score ≤ 1.0
- `behavioral_signals` added to evidence extraction output
- Communication renamed to "Communication & Behavioral" with behavioral rubric
- 0 / 0.5-1 sub-bands added to all rubrics for finer-grained low-end scoring

### Test Results

| Transcript | v1.0 Decision | v1.1 Decision | Ground Truth | v1.1 Better? |
|------------|---------------|---------------|--------------|--------------|
| transcript_1 | Reject | TBD | Borderline | TBD |
| Manas .NET Lead | TBD | TBD | TBD | TBD |

---

## v1.0 (2025-05-16)

### Status: Baseline

### Changes
- Initial implementation of all 4 prompts

### Test Results

| Transcript | Decision | Ground Truth | Match | Issues Caught |
|------------|----------|--------------|-------|---------------|
| transcript_1 | Reject | Borderline | Partial | Most issues caught, decision off by one band |

### Notes
- Decision on transcript_1 was Reject vs ground truth Borderline — strong Coding (4.5) and Fundamentals (4.0) were not enough to compensate for Architecture (2.5) at lead threshold 4.0
- Behavioral signals not captured at all

---

## Template for future versions

## vX.Y (YYYY-MM-DD)

### Changes
- **evidence_extraction.txt**: [What changed and why]
- **dimension_scoring.txt**: [What changed and why]

### Reason
[Which test transcripts failed and what pattern was observed]

### Results

| Transcript | vX.Y-1 Score | vX.Y Score | Ground Truth | Improvement? |
|------------|--------------|------------|--------------|--------------|
| #1 | 3.2 | 3.2 | Reject | Same |
| #3 | 3.8 (miss) | 3.5 (hit) | Reject | Yes |

### Decision
[Promote to baseline / needs more testing / revert]
