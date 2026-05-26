# Prompt Changelog

## v1.3 — post-release fixes (2026-05-19)

### Bug fix: prompt example quotes copied into real evaluations
The model was copying verbatim example quotes from the `evidence_extraction.txt` output format section into real evaluations. Detected by the hallucination gate: `"I designed the event-driven architecture using Kafka"` and `"We used microservices at my company"` appeared as ownership signals for Gautam (a .NET Lead candidate with no Kafka background), and `"I haven't actually worked with microservices in my current role"` appeared for Athira. Both are fictional examples from the prompt's JSON format illustration.

**Fix:** All three `technical_statements` examples and both `ownership_signals` examples replaced with domain-neutral illustrative content that cannot appear verbatim in real .NET or QA transcripts.

### New: hallucination detection
- `scripts/measure_accuracy.py` — `check_hallucinations()` verifies every extracted quote against the source transcript using 4-word n-gram matching. Results shown as a separate gate in the accuracy report; batch table gains a "Quotes" column.
- `scripts/run_evaluation.py` — `_warn_hallucinations()` fires automatically after every live evaluation, printing any unverifiable quotes before the summary.

### New: STT-tolerant pattern matching
`scripts/measure_accuracy.py` — `_fuzzy_match()` replaces exact substring matching for `quote_patterns` in ground truth checks. Uses `SequenceMatcher` sliding-window at threshold 0.75 to catch 1–2 character STT substitutions (e.g. "injection" → "ingestion", "singleton patern" missing a letter).

### New: descriptive output filenames
`scripts/run_evaluation.py` — output files now use a human-readable stem derived from the transcript filename and role level, e.g. `L1_Interview_Nimil_E_Raveendran_Tech_Lead_Net_lead_2dd89391.md`. Previously all outputs were `evaluation_{id}.md`.

---

## v1.3 (2026-05-19)

### Changes
- **evidence_extraction.txt**: Added `explanation_quality` field to every technical statement — classifies each as `correct_unprompted`, `correct_prompted`, `partially_correct`, `incorrect`, or `vague_no_depth`. Incorrect answers must be extracted as weaknesses.
- **evidence_extraction.txt**: Added `coding_required_significant_guidance` and `scenario_required_significant_guidance` flags to `problem_solving_approach` — captures whether the candidate needed repeated interviewer prompting to reach each step.
- **evidence_extraction.txt**: Added `communication_quality` object — `overall` (clear_structured / adequate / fragmented / confused) and `required_heavy_prompting` boolean.
- **dimension_scoring.txt**: Added "Vocabulary ≠ Competence" global rule — naming a concept without a correct explanation scores the same as not covering it. Incorrect answers lower the score, not just fail to raise it.
- **dimension_scoring.txt**: Added "Prompted vs Proactive" global rule — `scenario_required_significant_guidance: true` caps Architecture at ≤ 3.0; `coding_required_significant_guidance: true` caps Coding at ≤ 2.5.
- **dimension_scoring.txt**: Added "Heavy prompting lowers Communication" global rule — `required_heavy_prompting: true` caps Communication ≤ 2.5.
- **dimension_scoring.txt**: Coding rubric revised — score 1.0 now means "not assessed in this interview" (neutral, not a failure). Score 0.5 = attempted with critical failures. Score 0 = refused or produced nothing.
- **decision_logic.py**: Coding red flag threshold lowered from `< 2.0` to `< 1.0` — "not assessed" (1.0) no longer triggers auto-reject.
- **decision_logic.py**: "Borderline" renamed to "Conditional". Conditional band expanded from `threshold - 0.3` to `threshold - 2.0` with at least 2 dimensions ≥ 3.0. This captures genuine borderline profiles that have real strengths in key areas.

### Reason
v1.2 evaluations on 4 new transcripts (Athira, Nimil, Gautam, Shaik) showed a consistent failure mode: the model detects topic vocabulary and credits it as competence without testing explanation quality. Fundamentals scores were inflated 1.5–2.0 points above ground truth across all candidates. The Conditional decision band was never reached because: (a) Coding < 2.0 auto-rejected all candidates who weren't given coding tasks, and (b) the Borderline window (threshold - 0.3) was too narrow for borderline profiles where correct scores sit 1.5–2.0 below threshold.

### Test Comparison
| Transcript | v1.2 Accuracy | v1.3 | Better? |
|------------|--------------|------|---------|
| Manas (.NET Lead) | 98.0% | 97.9% | ✓ Stable |
| Athira (.NET Lead) | 58.6% | 97.9% | ✓ +39.3pp |
| Nimil (.NET Lead) | 22.9% | 97.9% | ✓ +75.0pp |
| Gautam (.NET Lead) | 69.5% | 91.2% | ✓ +21.7pp |
| Shaik (Sr QA Auto) | 40.4% | 95.0% | ✓ +54.6pp |

**Batch average: 91.5% across 7 evaluations — Phase 0 accuracy target MET.**

**Remaining miss (Gautam):** `design_pattern_explanations_generic_confused` — model extracted Repository/Strategy/CQRS as strengths (`correct_prompted`), missing that factory pattern explanation was confused and overall depth was generic. Factory pattern not independently extracted from the 65-minute transcript. Does not affect pass/fail for this candidate.

**Remaining miss (Nimil, earlier run):** `architecture_required_repeated_probing` — signal is interviewer behavioral pattern (repeated redirects), not candidate text. Latest run catches this via `scenario_required_significant_guidance` flag.

---

## v1.2 (2026-05-18)

### Changes
- **evidence_extraction.txt**: Split `problem_solving_approach` into separate coding and scenario tracks. Added `coding_attempted` flag — scenario/design answers no longer conflated with coding attempts.
- **evidence_extraction.txt**: Explicit guidance that scenario questions ("how would you build X") are NOT coding evidence — they test Architecture and consulting mindset.
- **evidence_extraction.txt**: Screen-share coding question detection — "let me share a piece of code" + line number references + exception names = `coding_attempted: true` even when code is not in transcript.
- **evidence_extraction.txt**: Ownership detection rewritten — "we" vs "I" is NOT the signal. The signal is whether the candidate can articulate their specific personal contribution when directly probed. Using "we" to credit the team is healthy; "we" that masks inability to separate personal role is exposure.
- **dimension_scoring.txt**: Added CRITICAL note to Coding rubric: scenario/design answers must NOT be scored here. Score 0.5-1 if no actual coding occurred; note it explicitly.
- **dimension_scoring.txt**: Added "Scored from" and consulting mindset signals to Architecture rubric.
- **dimension_scoring.txt**: Ownership rubric rewritten — clarifies that "we" is not automatically negative; scores on ability to articulate personal decisions and contributions when asked.
- **output_naresh.txt / output_quick_summary.txt**: Removed code fence wrappers from structure templates so output renders as clean markdown.

### Reason
v1.1 evaluation (Manas / .NET Lead) scored Coding at 2.5 because the model treated a scenario question ("walk me through building a login feature") as a coding attempt. The actual coding question (exception handling trace) was correctly identified but the login scenario inflated the Coding score. Scenario questions belong under Architecture; the Coding dimension should only reflect actual write/trace/debug tasks.

### Test Comparison
| Transcript | v1.1 | v1.2 | Better? |
|------------|------|------|---------|
| Manas (.NET Lead) | Coding: 2.5 | TBD | Run to verify |

---

## v1.1 (2026-05-17)

### Changes
- **dimension_scoring.txt**: Added global "no-experience floor" rule — admitted no experience = score ≤ 1.0 (v1.0 was defaulting these to 2.0)
- **dimension_scoring.txt**: Renamed "Communication" to "Communication & Behavioral" with explicit behavioral rubric covering conflict resolution, feedback giving, cross-functional collaboration, and handling ambiguity
- **dimension_scoring.txt**: Added granular 0 / 0.5-1 sub-bands to all rubrics so the model doesn't skip 0-1 when evidence is clearly absent
- **evidence_extraction.txt**: Added `behavioral_signals` extraction — captures conflict/disagreement handling, feedback approach, collaboration style, communication with stakeholders
- **evidence_extraction.txt**: Added explicit rule to extract no-experience admissions as weaknesses (not ignore them)
- **output_naresh.txt**: Added Communication & Behavioral line to structure template

### Reason
First production evaluation (Manas / .NET Lead) scored no-experience skills at 2.0 instead of ≤1.0, making the candidate appear more capable than they were. Behavioral signals (communication style, conflict handling) were not captured at all despite being critical for a Lead role assessment.

### Test Comparison
| Transcript | v1.0 | v1.1 | Ground Truth | Better? |
|------------|------|------|--------------|---------|
| Manas (.NET Lead) | TBD | TBD | TBD | Run to verify |

---

## v1.0 (2025-05-16)

### Initial release

**Files:**
- `evidence_extraction.txt` - Extract technical statements, ownership signals, buzzwords, problem-solving approach
- `dimension_scoring.txt` - Score 7 dimensions (Fundamentals, Coding, Architecture, Ownership, Communication, Practical, Learning) on 0-5 scale with rubrics
- `output_naresh.txt` - Generate direct, evidence-backed feedback in Naresh's style (300-500 words)
- `output_quick_summary.txt` - Generate recruiter-friendly 150-250 word summary with ✓/✗ key points

### Design decisions

- Evidence extraction is a separate AI call from scoring to avoid anchoring bias in evidence extraction
- Ownership vs exposure distinction is explicit in both extraction and scoring prompts
- Buzzword detection is integrated into evidence extraction
- Decision logic is deterministic (rule-based) to ensure consistency - no AI in decision step
- Two output formats: detailed (Naresh-style) and quick (recruiter-friendly)
