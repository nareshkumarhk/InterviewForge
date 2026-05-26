# Test Transcripts

This directory contains interview transcripts used for evaluating the agent.

## File Format

Each test case requires three files:

```
{stem}.txt              # Raw interview transcript
{stem}_metadata.json    # Role and interview context
{stem}_groundtruth.json # Human expert evaluation for accuracy measurement
```

## Transcript Format (`{stem}.txt`)

Plain text with interviewer/candidate turns:

```
Interviewer: [Question]

Candidate: [Answer]
```

Keep the format consistent. Timestamps are optional.

## Metadata Format (`{stem}_metadata.json`)

```json
{
  "role": {
    "role_title": "Senior Backend Developer",
    "role_level": "senior",           // junior | intermediate | senior | lead
    "experience_years": "6-10",
    "must_have_skills": [
      {
        "skill": "Python",
        "depth": "expert",            // awareness | working | proficient | expert
        "importance": "critical",     // critical | important | moderate
        "context": "Production-scale async processing"
      }
    ],
    "nice_to_have": ["Redis", "Kafka"],
    "key_responsibilities": [
      "Design backend services",
      "Lead architecture decisions"
    ]
  },
  "interview": {
    "round_type": "L1_Technical",
    "interviewer": "Bilal",
    "duration_minutes": 50,
    "date": "2025-05-15"
  }
}
```

## Ground Truth Format (`{stem}_groundtruth.json`)

```json
{
  "candidate_name": "Jane Smith",
  "role": "dotnet_lead",
  "interview_date": "2026-05-18",
  "interviewer": "Naresh Kumar",

  "expected_decision": "Reject",
  "expected_calibration": "One paragraph summary of why — what was strong, what was weak, what was the overall signal",

  "expected_dimension_ranges": {
    "Fundamentals":               { "min": 1.5, "max": 2.5 },
    "Coding":                     { "min": 0.5, "max": 1.0 },
    "Architecture":               { "min": 1.0, "max": 2.0 },
    "Ownership":                  { "min": 2.0, "max": 3.0 },
    "Communication & Behavioral": { "min": 1.5, "max": 2.5 },
    "Practical":                  { "min": 1.5, "max": 2.5 },
    "Learning":                   { "min": 2.0, "max": 3.0 }
  },

  "critical_issues": [
    {
      "id": "short_snake_case_id",
      "description": "Full description of the issue for display in reports",
      "quote_patterns": ["exact phrase", "variant phrase"],
      "topic_patterns": ["Topic Name", "Alt Topic Name"],
      "expected_evidence_type": "weakness",
      "dimension": "Fundamentals",
      "severity": "critical",
      "must_catch": true
    }
  ],

  "known_strengths": [
    {
      "id": "short_snake_case_id",
      "description": "What the strength is",
      "quote_patterns": ["phrase from transcript"],
      "topic_patterns": ["Topic Name"],
      "expected_evidence_type": "strength",
      "dimension": "Fundamentals"
    }
  ],

  "scoring_notes": {
    "key_pattern": "What systematic failure this ground truth guards against"
  },

  "notes": "Free-text context about the interview — duration, format, anything unusual"
}
```

**Decision values:** `"Strong Hire"` | `"Hire"` | `"Conditional"` | `"Reject"`

**Issue detection logic** (in `scripts/measure_accuracy.py`):
- An issue is caught if its `quote_patterns` appear (fuzzy-matched) in any extracted quote, OR its `topic_patterns` appear in any extracted topic, OR relevant flags match in `problem_solving_approach` or `communication_quality`.
- Issues with `"must_catch": true` count toward the critical issues score.
- `check_flag` / `expected_flag_value` can also be used to match boolean flags in `problem_solving_approach`.

## Accuracy Measurement

Run `scripts/measure_accuracy.py` to score an evaluation against its ground truth:

```bash
# Single evaluation:
python scripts/measure_accuracy.py --result tests/results/v1.3/evaluation_abc.json

# All evaluations in a version:
python scripts/measure_accuracy.py --results-dir tests/results/v1.3
```

An evaluation passes (≥ 90%) when: decision matches (40%), must-catch issues detected (35%), dimension scores in range (15%), known strengths found (10%). The hallucination gate is shown separately — it verifies every extracted quote traces back to the transcript.

## Adding New Test Cases

1. Place transcript in `tests/transcripts/` — use the recording filename as-is
2. Fill in accurate metadata from the JD/role brief (`_metadata.json`)
3. Write ground truth **before** running the AI — anchoring bias is real (`_groundtruth.json`)
4. Run evaluation: `python scripts/run_evaluation.py --transcript ... --version v1.3`
5. Score accuracy: `python scripts/measure_accuracy.py --result tests/results/v1.3/evaluation_xxx.json`
6. Document results in `prompts/CHANGELOG.md` if you made prompt changes
