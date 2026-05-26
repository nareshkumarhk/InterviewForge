# Interview Review Agent - Phase 0

AI-powered interview evaluation engine that analyzes technical interview transcripts and generates structured hiring feedback.

**Phase 0 goal:** Validate that AI can generate interview evaluations that hiring managers trust and use. Success = 90%+ accuracy across test transcripts.

---

## Setup

### 1. Python environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API key

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## Usage

### Run an evaluation

```bash
python scripts/run_evaluation.py \
  --transcript "tests/transcripts/MyTranscript.txt" \
  --version v1.3
```

Options:
- `--version v1.3` — prompt version (default: v1.0; use latest: v1.3)
- `--output path/to/output` — custom output path
- `--interviewer-analysis` — include interviewer quality analysis (Phase 1+)

Output is saved to `tests/results/v1.3/` using a descriptive filename derived from the transcript name and role level, e.g.:

```
L1_Interview_Nimil_E_Raveendran_Tech_Lead_Net_lead_2dd89391.md
L1_Interview_Nimil_E_Raveendran_Tech_Lead_Net_lead_2dd89391.json
```

Hallucination warnings are printed automatically if any extracted quotes cannot be traced to the transcript.
If the API rate limit is hit mid-evaluation, a partial result is saved to `tests/results/partial/` so progress is not lost.

### Measure accuracy against ground truth

```bash
# Single evaluation:
python scripts/measure_accuracy.py --result tests/results/v1.3/evaluation_abc.json

# All evaluations in a version:
python scripts/measure_accuracy.py --results-dir tests/results/v1.3
```

Reports decision accuracy, critical issue detection rate, score calibration, and quote verification (hallucination gate).

### Compare prompt versions

```bash
# Run fresh evaluations for each version (makes API calls):
python scripts/compare_versions.py \
  --transcript "tests/transcripts/MyTranscript.txt" \
  --versions v1.2 v1.3

# Compare existing result files (no API calls):
python scripts/compare_versions.py \
  --results tests/results/v1.2/evaluation_abc.json tests/results/v1.3/evaluation_def.json
```

Shows a side-by-side score and decision comparison across versions. Add `--save` to persist new evaluation results to disk, or `--reuse` to load the most recent saved result for each version instead of calling the API again.

### Check API rate limits

```bash
python scripts/check_rate_limits.py
python scripts/check_rate_limits.py --model gpt-4o-mini
```

Makes a 1-token call and reports remaining requests/tokens and reset times. Useful before running a batch of evaluations.

---

## Project structure

```
interview-review-agent/
├── config/
│   ├── role_definitions.json     # Predefined role templates (dotnet_lead, senior_qa_automation, …)
│   └── scoring_rubrics.json      # Scoring rubrics and thresholds
├── prompts/
│   ├── v1.3/                     # Current prompt version
│   │   ├── evidence_extraction.txt
│   │   ├── dimension_scoring.txt
│   │   ├── output_naresh.txt
│   │   └── output_quick_summary.txt
│   └── CHANGELOG.md              # Prompt change history and accuracy results
├── src/
│   ├── engine.py                 # Orchestrates evaluation pipeline
│   ├── ai_client.py              # OpenAI API wrapper with rate-limit tracking
│   ├── prompts.py                # Prompt loading and versioning
│   ├── decision_logic.py         # Rule-based hiring decision (no AI)
│   ├── transcript_compressor.py  # Two-stage transcript compression (rules → AI)
│   └── models.py                 # Pydantic data models
├── tests/
│   ├── transcripts/              # Test transcripts + metadata + ground truth
│   │   └── README.md             # File format reference
│   └── results/                  # Evaluation outputs per version
│       └── v1.3/
├── scripts/
│   ├── run_evaluation.py         # Main evaluation CLI
│   ├── measure_accuracy.py       # Accuracy measurement vs ground truth + hallucination gate
│   ├── compare_versions.py       # Side-by-side version comparison
│   └── check_rate_limits.py      # Check OpenAI rate limit headroom before a batch run
└── docs/
    └── PROMPT_ITERATION_LOG.md   # Redirects to prompts/CHANGELOG.md
```

---

## Evaluation pipeline

```
Transcript + Role Metadata
         │
         ▼
0. Transcript Compression (rule-based, then AI if still over budget)
   - Stage 1: strips timestamps, filler words, ack-only lines
   - Stage 2: AI compression pass (only if stage 1 > 26k tokens)
         │
         ▼
1. Evidence Extraction (AI)
   - Technical statements: topic, quote, depth, evidence_type,
     explanation_quality (correct_unprompted / correct_prompted /
     partially_correct / incorrect / vague_no_depth)
   - Ownership vs exposure signals
   - Problem-solving approach: coding vs scenario, guidance required
   - Communication quality: overall rating + required_heavy_prompting flag
   - Buzzword detection
         │
         ▼
2. Dimension Scoring (AI)
   - 7 dimensions: Fundamentals, Coding, Architecture,
     Ownership, Communication & Behavioral, Practical, Learning
   - 0-5 scale with decimal precision
   - Global rules: vocabulary ≠ competence, prompted vs proactive,
     heavy prompting caps Communication ≤ 2.5
         │
         ▼
3. Hiring Decision (Rule-based, deterministic)
   - Weighted score by role level
   - Red flag auto-reject checks
   - Strong Hire / Hire / Conditional / Reject
         │
         ▼
4a. Naresh-style Feedback (AI)    4b. Quick Summary (AI)
    300-500 words, direct              150-250 words
    evidence-backed                    recruiter-friendly
         │
         ▼
5. Hallucination Check (deterministic)
   - Every extracted quote verified against source transcript
   - 4-word n-gram matching with STT-error tolerance
   - Flagged quotes printed as warnings; gate shown in accuracy report
```

---

## Prompt versioning workflow

1. **Identify failure pattern** — run `measure_accuracy.py` against ground truth, inspect missed issues
2. **Edit prompts** — create `prompts/v1.4/` by copying `v1.3/` and modifying
3. **Run comparison** — `python scripts/compare_versions.py --versions v1.3 v1.4`
4. **Document results** — update `prompts/CHANGELOG.md` with accuracy before/after
5. **Promote** — if v1.4 is better, pass `--version v1.4` when running evaluations

---

## Adding test transcripts

1. Place transcript in `tests/transcripts/` (any filename, `.txt`)
2. Create metadata: `tests/transcripts/{stem}_metadata.json`
3. Write ground truth **before** running AI: `tests/transcripts/{stem}_groundtruth.json`
4. See `tests/transcripts/README.md` for exact field schemas

---

## Interpreting results

### Dimension scores (0-5)

| Score | Meaning |
|-------|---------|
| 0-1 | Cannot perform - basic gaps |
| 2 | Exposure/surface level |
| 3 | Solid, independent performance |
| 4 | Strong, above average |
| 5 | Expert, teaches others |

### Hiring thresholds

| Role Level | Threshold |
|------------|-----------|
| Junior | 3.0 |
| Intermediate | 3.5 |
| Senior | 4.0 |
| Lead | 4.2 |

### Decision bands

| Decision | Condition |
|----------|-----------|
| Strong Hire | Weighted score ≥ threshold + 0.5 |
| Hire | Weighted score ≥ threshold |
| Conditional | Weighted score ≥ threshold − 2.0 (candidate has real substance but gaps; recommend follow-up round) |
| Reject | Weighted score < threshold − 2.0, or a red flag triggered |

### Red flags (automatic Reject)

- Fundamentals < 2.0 for intermediate/senior/lead roles
- Coding < 1.0 — only triggers when coding was attempted and produced nothing (score 1.0 means coding was not assessed, which is not a failure)
- Ownership < 2.0 for senior/lead roles

---

## Phase 0 success criteria

- [x] 90%+ accuracy across 5-10 test transcripts — **91.5% avg across 7 evaluations (v1.3)**
- [x] Zero critical hallucinations — **hallucination gate added; quote verification runs on every evaluation**
- [ ] Hiring manager validation: "I would use this"
- [x] All decisions have evidence-backed reasoning — **every decision includes weighted score, threshold, and red flag reason if triggered**
