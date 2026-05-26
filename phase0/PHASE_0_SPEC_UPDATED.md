# Interview Review Agent — Phase 0 Specification (Updated)

> **Original spec:** `PHASE_0_SPEC_1.md` (May 2025)
> **Updated:** May 2026 — reflects what was built, what changed, and what remains pending

---

## Project Overview

An AI-powered interview evaluation engine that analyzes technical interview transcripts and generates structured hiring feedback. Phase 0 (Proof of Value) — Python CLI only, no infrastructure, focused entirely on AI prompt quality and evaluation accuracy.

**Goal:** Validate that AI can generate interview evaluations that hiring managers trust and use.

**Success Criteria:** 90% accuracy across 5–10 test transcripts (AI catches 90%+ of critical issues that humans caught, with no major fabrications).

---

## What Changed from the Original Spec

### Added (not in original spec)

| Feature | Where | Why |
|---|---|---|
| Rate limit visibility | `src/ai_client.py` | TPM limit hit in first production run; need to know when to retry |
| `RateLimitExhausted` exception | `src/ai_client.py` | Distinguish billing failure (fatal) from rate limit (retryable) |
| Partial result saving | `src/engine.py` | Save completed steps if rate limit hits mid-evaluation |
| `scripts/check_rate_limits.py` | `scripts/` | One-command probe to see current OpenAI rate limit state |
| Transcript compression module | `src/transcript_compressor.py` | Teams transcripts are 100K+ tokens — 4× over TPM limit without compression |
| Token budget management | `src/engine.py` | Configurable via `TRANSCRIPT_TOKEN_BUDGET` env var |
| Behavioral signals extraction | prompts `v1.1+` | Behavioral dimensions (conflict, feedback, collaboration) not captured in v1.0 |
| No-experience scoring floor | prompts `v1.1+` | v1.0 defaulted gaps to 2.0; correct floor is ≤1.0 |
| Coding vs scenario disambiguation | prompts `v1.2+` | Scenario questions scored under Coding by mistake in v1.1 |
| Screen-share coding recognition | prompts `v1.2+` | Code shared on screen doesn't appear in transcript; need signal detection |
| Multi-form coding question types | prompts `v1.2+` | Trace/predict/debug/review all count as coding — not just write |
| `dotnet_lead` role definition | `config/role_definitions.json` | First production evaluation (Manas / .NET Lead) |
| `--interviewer-analysis` CLI flag | `scripts/run_evaluation.py` | Optional step; excluded from Phase 0 default |

### Modified from spec

| Item | Spec | Current |
|---|---|---|
| Default model | `gpt-4-turbo-preview` (retired) | `gpt-4o` (or `OPENAI_MODEL` env var override) |
| Retry strategy | 3 retries, linear 2/4/6s | 5 retries, exponential 5/10/20/40s |
| "Communication" dimension | `Communication` | `Communication & Behavioral` (includes conflict, feedback, collaboration) |
| `problem_solving_approach` | Generic dict | Structured with `coding_*` and `scenario_*` tracks |
| Evidence model | No behavioral signals | `behavioral_signals` field added |
| Output prompts | Template wrapped in code fences | Code fences removed — MD renders clean |

### Pending from original spec (not yet built)

| Item | Status | Notes |
|---|---|---|
| `scripts/compare_versions.py` | Not built | A/B comparison across prompt versions |
| Ground truth files (`_groundtruth.json`) | Not built | Required for 90% accuracy measurement |
| Accuracy metrics tracking | Not built | Needed to meet Phase 0 success criteria |
| `config/scoring_rubrics.json` | Not built | Rubrics live in prompts instead; may not be needed |
| Interviewer analysis (AI Call #3) | Stub only | Intentionally deferred to Phase 1 |

---

## Project Structure (Current)

```
interview-review-agent/
├── README.md
├── requirements.txt
├── .env                          # OPENAI_API_KEY, OPENAI_MODEL (gitignored)
├── .env.example
├── config/
│   ├── role_definitions.json     # Role profiles: senior_backend, dotnet_lead, ...
│   └── scoring_rubrics.json      # (stub — rubrics live in prompts)
├── prompts/
│   ├── v1.0/                     # Initial prompts
│   ├── v1.1/                     # No-experience floor + behavioral signals
│   ├── v1.2/                     # Coding/scenario split + screen-share detection
│   └── CHANGELOG.md
├── src/
│   ├── __init__.py
│   ├── engine.py                 # Core evaluation engine + transcript fitting
│   ├── prompts.py                # Prompt version management
│   ├── ai_client.py              # OpenAI wrapper: retries, rate limits, error surfacing
│   ├── transcript_compressor.py  # NEW: two-stage transcript compression
│   ├── decision_logic.py         # Rule-based scoring (deterministic, no AI)
│   └── models.py                 # Pydantic models
├── tests/
│   ├── transcripts/
│   │   ├── README.md
│   │   ├── <name>.txt            # Transcript file
│   │   └── <name>_metadata.json  # Role + interview metadata
│   └── results/
│       ├── v1.0/
│       ├── v1.1/
│       ├── v1.2/
│       └── partial/              # NEW: partial saves on rate-limit failure
├── scripts/
│   ├── run_evaluation.py         # Main CLI
│   ├── check_rate_limits.py      # NEW: rate limit probe
│   └── compare_versions.py       # PENDING: A/B version comparison
└── docs/
    └── PROMPT_ITERATION_LOG.md
```

---

## Technical Stack

- **Python:** 3.10+
- **AI Service:** OpenAI API — `gpt-4o` default, `gpt-4o-mini` via `OPENAI_MODEL` for higher TPM
- **Key Libraries:** `openai`, `pydantic`, `python-dotenv`, `rich`

**No Infrastructure:** No databases, no APIs, no cloud services. CLI only.

---

## Core Components

### 1. AI Client (`src/ai_client.py`)

Wrapper around OpenAI API with full observability into rate limit state.

**Key additions beyond spec:**

```python
class RateLimitExhausted(AIClientError):
    """Rate limit persists after all retries. Carries current limit state."""
    def __init__(self, message: str, rate_limit: Optional[RateLimitInfo] = None): ...

@dataclass
class RateLimitInfo:
    limit_requests: Optional[int]
    remaining_requests: Optional[int]
    limit_tokens: Optional[int]
    remaining_tokens: Optional[int]
    reset_requests: Optional[str]   # e.g. "1s", "6m0s"
    reset_tokens: Optional[str]
```

- Uses `.with_raw_response` on every API call to capture `x-ratelimit-*` headers
- Exponential backoff: 5s, 10s, 20s, 40s (5 attempts total)
- `insufficient_quota` (billing error) → fail fast, no retry
- Rate limit error → prints exact API error body (`code`, `type`, `message`), then retries

---

### 2. Transcript Compressor (`src/transcript_compressor.py`) — NEW

Two-stage compression for Microsoft Teams `.ordered.transcript` format.

**Why needed:** Teams transcripts encode each utterance 5–6 times in different format variants. A 36-minute interview = 170K tokens raw. After deduplication + cleaning = ~18K tokens. Well within the 26K budget.

**Stage 1 — Rule-based (always runs):**

```
Pipeline: _strip_noise → _split_into_turns → _remove_filler_and_acks → _dedupe_lines
```

| Step | What it removes | Token impact |
|---|---|---|
| `_strip_noise` | Unicode PUA chars (Teams icons), system events | Minor |
| `_split_into_turns` | Timestamp noise, repeated speaker headers; converts to `Speaker: content` | Large |
| `_remove_filler_and_acks` | um/uh/basically, "Yeah."/"Okay." lines | Minor |
| `_dedupe_lines` | Exact duplicate lines from multi-format Teams encoding | Critical — 80%+ reduction |

**Teams new-turn formats handled:**
- Format A: `Name H:MM Name X minutes Y seconds` (first turn / after system event)
- Format B: `Name X minutes Y seconds H:MM Name X minutes Y seconds` (subsequent turns)
- Continuation: `Name X minutes Y seconds` (same speaker, continued)

**Stage 2 — AI compression (only if stage 1 > budget):**

- Splits into ~20K-token chunks (respects gpt-4o-mini 128K context window)
- Compresses each chunk proportionally, reassembles
- Preserves: all technical claims, ownership language, technology names, behavioral signals, code discussed

**Token budget:**
- Default: 26,000 tokens (leaves ~4K headroom for system prompt + metadata)
- Override: `TRANSCRIPT_TOKEN_BUDGET` env var

---

### 3. Evaluation Engine (`src/engine.py`)

Orchestrates the 5-step pipeline. Key additions:

**Transcript fitting before evaluation:**
```python
def _fit_transcript(self, transcript: str, evaluation_id: str) -> Tuple[str, int]:
    # Stage 1: rule-based (always)
    # Stage 2: AI chunked compression (only if still over budget)
```

**Rate limit visibility after each step:**
```python
def _print_rate_limit(self, evaluation_id: str, step: str) -> None:
    # Prints: requests X/Y | tokens X/Y | resets in Xs
```

**Partial result saving on `RateLimitExhausted`:**
```python
def _save_partial(self, ...) -> Path:
    # Saves to tests/results/partial/partial_{id}.json
    # Includes: completed steps, evidence, scores, decision (whatever completed)
```

---

### 4. Decision Logic (`src/decision_logic.py`)

Unchanged from spec. Deterministic, no AI.

**Thresholds:** Junior 3.0 / Intermediate 3.5 / Senior 4.0 / Lead 4.2

**Dimension weights by role level** (Lead example):
```
Architecture: 30%  |  Ownership: 25%  |  Fundamentals: 15%
Practical: 15%     |  Communication: 10%  |  Coding: 5%
```

**Automatic Reject red flags:**
- Fundamentals < 2.0 for Intermediate/Senior/Lead
- Coding < 2.0 for any engineering role
- Ownership < 2.0 for Senior/Lead

---

### 5. Data Models (`src/models.py`)

Additions to original spec:

```python
class Evidence(BaseModel):
    technical_statements: List[EvidenceItem]
    ownership_signals: List[OwnershipSignal]
    problem_solving_approach: Optional[dict] = None   # structured: coding_* + scenario_* tracks
    behavioral_signals: List[dict] = []               # NEW: conflict, feedback, collaboration
    buzzwords_flagged: List[str] = []
```

---

## Prompt Versions

### v1.0 — Initial release

7 dimensions: Fundamentals, Coding, Architecture, Ownership, Communication, Practical, Learning.
Basic evidence extraction with ownership/exposure detection and buzzword flagging.

### v1.1 — Scoring accuracy fixes

**Problem:** First production run (Manas / .NET Lead) gave 2.0 to skills the candidate explicitly had no experience with.

**Changes:**
- Global no-experience floor: explicit admission = score ≤ 1.0
- `Communication` renamed to `Communication & Behavioral`
- New behavioral rubric: conflict handling, feedback giving, cross-functional collaboration
- `behavioral_signals` extraction field added to evidence
- Granular 0 / 0.5–1 sub-bands added to all rubrics
- Explicit rule: no-experience admissions must be extracted as weaknesses

### v1.2 — Coding dimension accuracy + format fixes

**Problems:** (a) Scenario questions ("walk me through building a login system") were being scored under Coding instead of Architecture. (b) Screen-shared code trace questions weren't recognized. (c) Output MD files had spurious code fences.

**Changes:**
- Coding rubric: CRITICAL note — scenario questions ≠ coding. Only write/trace/predict/debug/review count.
- Full list of coding question forms: write, trace execution, predict output, execution flow, debug, code review, refactor
- Screen-share coding detection: "let me share a piece of code" + line number references + exception names = `coding_attempted: true`
- `problem_solving_approach` split into `coding_*` and `scenario_*` tracks
- Architecture rubric: explicit "Scored from" guidance — scenario questions, requirement clarification, trade-off identification feed here
- Architecture rubric: consulting mindset signals added for lead-level (clarifies requirements, surfaces non-functional concerns, identifies trade-offs)
- Output prompt templates: removed code fence wrappers → clean MD rendering

---

## Usage

### Run an evaluation

```bash
python scripts/run_evaluation.py \
  --transcript tests/transcripts/my_transcript.txt \
  --version v1.2
```

Required alongside transcript:
- `<transcript_name>_metadata.json` — role + interview metadata (same folder)

Optional flags:
- `--output path/to/output` — override default output path
- `--interviewer-analysis` — enable optional interviewer quality analysis (Phase 1+)

### Check rate limits

```bash
python scripts/check_rate_limits.py
python scripts/check_rate_limits.py --model gpt-4o-mini
```

### Token budget override (for larger transcripts)

```bash
TRANSCRIPT_TOKEN_BUDGET=20000 python scripts/run_evaluation.py --transcript ...
```

### Use a different model

```bash
# In .env:
OPENAI_MODEL=gpt-4o-mini   # higher TPM, lower cost, slightly lower quality
OPENAI_MODEL=gpt-4o        # default, higher quality
```

---

## Metadata Format

### `<transcript>_metadata.json`

```json
{
  "role": {
    "role_title": ".NET Lead / Backend Lead",
    "role_level": "lead",
    "experience_years": "8-12",
    "must_have_skills": [
      {
        "skill": ".NET / C#",
        "depth": "expert",
        "importance": "critical",
        "context": "Production backend services"
      }
    ],
    "nice_to_have": ["Azure", "Microservices"],
    "key_responsibilities": [
      "Lead backend architecture decisions",
      "Mentor team members",
      "Own production systems"
    ]
  },
  "interview": {
    "round_type": "L1_Technical",
    "interviewer": "Naresh Kumar",
    "duration_minutes": 60,
    "date": "2026-05-15"
  }
}
```

`role_level` values: `junior` | `intermediate` | `senior` | `lead`
`depth` values: `awareness` | `working` | `proficient` | `expert`
`importance` values: `critical` | `important` | `moderate`

---

## Output Files

For each evaluation, two files are saved to `tests/results/{version}/`:

**`evaluation_{id}.json`** — Full structured result:
- Evaluation ID, timestamp, processing time
- Role and interview metadata
- Extracted evidence (technical statements, ownership signals, behavioral signals)
- 7 dimension scores with reasoning and evidence quotes
- Decision: recommendation + weighted score + threshold
- Both feedback formats (Naresh + quick summary)

**`evaluation_{id}.md`** — Naresh-style feedback, clean markdown, ready to share:
```markdown
# .NET Lead / Backend Lead Interview - L1_Technical

**Status:** Reject

## Remarks
...
## Interview Reliability Note
**Reliability:** High / Medium / Low
...
## Overall Take
...
## One-Line Summary
...
```

**`tests/results/partial/partial_{id}.json`** — Saved if rate limit hits mid-evaluation:
```json
{
  "status": "partial",
  "failed_at": "feedback_generation",
  "completed_steps": ["evidence_extraction", "dimension_scoring", "decision"],
  ...
}
```

---

## Acceptance Criteria (Phase 0)

| # | Criterion | Status |
|---|---|---|
| 1 | Core engine works: process transcript → evaluation | ✅ Done |
| 2 | All 4 AI calls implemented | ✅ Done |
| 3 | Decision logic works (rule-based) | ✅ Done |
| 4 | Prompt versioning works, can switch and compare | ✅ Done (v1.0 → v1.2) |
| 5 | Test suite: 5–10 transcripts with ≥90% accuracy | ⏳ Pending (1 transcript tested) |
| 6 | Output quality validated by hiring managers | ⏳ Pending |
| 7 | Documentation complete | ✅ Done |
| 8 | No hallucinations | ✅ Validated on current transcript |
| 9 | Evidence-backed: every claim has a supporting quote | ✅ Done |
| 10 | Ready for Gate 1 go/no-go | ⏳ Pending accuracy validation |

---

## Pending Work (to complete Phase 0)

### 1. Ground truth files and accuracy measurement

Create `_groundtruth.json` for each test transcript and a scoring script:

```json
{
  "actual_decision": "Reject",
  "actual_reasoning": "No microservices, no lead-level ownership, weak coding",
  "critical_issues_to_catch": [
    "Explicit no-microservices admission",
    "All ownership language is 'we' not 'I'",
    "Code trace question answered incorrectly"
  ],
  "expected_recommendation": "Reject",
  "expected_calibration": "Mid-level developer, not lead"
}
```

### 2. `scripts/compare_versions.py`

Run the same transcript through multiple prompt versions and produce a side-by-side comparison table. Useful for validating prompt improvements before promoting a version.

### 3. Expand test corpus

Need 4–9 more transcripts (across: Strong Hire, Hire, Borderline, Reject) to reach the 5–10 transcript target for accuracy measurement.

---

## Next Steps After Phase 0

**If Phase 0 succeeds (≥90% accuracy):**
1. Document lessons learned from prompt iteration
2. Promote v1.2 as baseline for Phase 1
3. Plan Phase 1 (MVP) architecture — web UI, database, multi-user
4. Budget approval for cloud infrastructure
5. Begin Phase 1 development

**If Phase 0 needs iteration:**
1. Analyze failure patterns from ground truth comparison
2. Refine prompts for specific failure modes
3. Re-test, repeat until 90% threshold met
