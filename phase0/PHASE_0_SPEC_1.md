# Interview Review Agent - Phase 0 Specification

## Project Overview

Build an AI-powered interview evaluation engine that analyzes technical interview transcripts and generates structured hiring feedback. This is Phase 0 (Proof of Value) - a Python-based evaluation engine with no infrastructure, focusing 100% on AI prompt quality and evaluation accuracy.

**Goal:** Validate that AI can generate interview evaluations that hiring managers trust and use.

**Success Criteria:** 90% accuracy across 5-10 test transcripts (AI catches 90%+ of critical issues that humans caught, with no major fabrications).

---

## Project Structure

```
interview-review-agent/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── role_definitions.json
│   └── scoring_rubrics.json
├── prompts/
│   ├── v1.0/
│   │   ├── evidence_extraction.txt
│   │   ├── dimension_scoring.txt
│   │   ├── output_naresh.txt
│   │   └── output_quick_summary.txt
│   └── CHANGELOG.md
├── src/
│   ├── __init__.py
│   ├── engine.py              # Core evaluation engine
│   ├── prompts.py             # Prompt management
│   ├── ai_client.py           # OpenAI API wrapper
│   ├── decision_logic.py      # Rule-based scoring
│   └── models.py              # Data models (Pydantic)
├── tests/
│   ├── transcripts/
│   │   ├── README.md          # Instructions for test data
│   │   ├── transcript_1.txt
│   │   ├── transcript_1_metadata.json
│   │   └── transcript_1_groundtruth.json
│   └── results/
│       └── v1.0/
│           └── .gitkeep
├── scripts/
│   ├── run_evaluation.py      # Main evaluation script
│   └── compare_versions.py    # Compare prompt versions
└── docs/
    └── PROMPT_ITERATION_LOG.md
```

---

## Technical Stack

- **Python:** 3.10+
- **AI Service:** OpenAI API (GPT-4 Turbo)
- **Key Libraries:**
  - `openai` - OpenAI API client
  - `pydantic` - Data validation and models
  - `python-dotenv` - Environment variable management
  - `rich` - Terminal output formatting
  - `pyyaml` or `json` - Configuration management

**No Infrastructure:** No databases, no APIs, no cloud services. Command-line execution only.

---

## Core Components

### 1. Data Models (src/models.py)

Define Pydantic models for type safety and validation:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class RoleLevel(str, Enum):
    JUNIOR = "junior"
    INTERMEDIATE = "intermediate"
    SENIOR = "senior"
    LEAD = "lead"

class Skill(BaseModel):
    skill: str
    depth: Literal["awareness", "working", "proficient", "expert"]
    importance: Literal["critical", "important", "moderate"]
    context: Optional[str] = None

class RoleMetadata(BaseModel):
    role_title: str
    role_level: RoleLevel
    experience_years: str  # e.g., "6-10"
    must_have_skills: List[Skill]
    nice_to_have: Optional[List[str]] = []
    key_responsibilities: List[str]

class InterviewMetadata(BaseModel):
    round_type: str  # "L1_Technical", "L2_System_Design"
    interviewer: str
    duration_minutes: int
    date: Optional[str] = None

class EvidenceItem(BaseModel):
    topic: str
    quote: str
    depth: Literal["superficial", "detailed", "expert"]
    evidence_type: Literal["strength", "weakness", "neutral"]
    context: Optional[str] = None

class OwnershipSignal(BaseModel):
    type: Literal["ownership", "exposure"]
    quote: str
    context: str

class Evidence(BaseModel):
    technical_statements: List[EvidenceItem]
    ownership_signals: List[OwnershipSignal]
    problem_solving_approach: Optional[dict] = None
    buzzwords_flagged: List[str] = []

class DimensionScore(BaseModel):
    name: str
    score: float = Field(..., ge=0, le=5)
    reasoning: str
    evidence: List[dict]

class InterviewerAnalysis(BaseModel):
    questioning_quality: float = Field(..., ge=0, le=5)
    topic_coverage: float = Field(..., ge=0, le=5)
    reliability_score: float = Field(..., ge=0, le=5)
    strengths: List[str]
    missed_opportunities: List[dict]
    overall_assessment: str

class Decision(BaseModel):
    recommendation: Literal["Strong Hire", "Hire", "Borderline", "Reject"]
    reasoning: str
    weighted_score: float
    threshold: float

class EvaluationResult(BaseModel):
    evaluation_id: str
    prompt_versions: dict
    transcript_file: str
    role: RoleMetadata
    interview: InterviewMetadata
    evidence: Evidence
    scores: List[DimensionScore]
    interviewer_analysis: Optional[InterviewerAnalysis] = None
    decision: Decision
    feedback_naresh: str
    feedback_quick: str
    processing_time_seconds: float
    timestamp: str
```

---

### 2. AI Client (src/ai_client.py)

Wrapper around OpenAI API with error handling and retry logic:

```python
from openai import OpenAI
import os
import time
from typing import Optional
import json

class AIClient:
    """
    OpenAI API client wrapper with retry logic and error handling.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_retries = 3
        self.retry_delay = 2  # seconds
    
    def complete(
        self, 
        system_prompt: str, 
        user_prompt: str,
        temperature: float = 0.1,
        response_format: Optional[dict] = None
    ) -> str:
        """
        Call OpenAI API with retry logic.
        
        Args:
            system_prompt: System instruction
            user_prompt: User message
            temperature: Sampling temperature (0-1)
            response_format: Optional {"type": "json_object"} for JSON mode
        
        Returns:
            Response text from model
        """
        for attempt in range(self.max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature
                }
                
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise Exception(f"API call failed after {self.max_retries} attempts: {e}")
    
    def complete_json(
        self, 
        system_prompt: str, 
        user_prompt: str,
        temperature: float = 0.1
    ) -> dict:
        """
        Call OpenAI API expecting JSON response.
        
        Returns:
            Parsed JSON object
        """
        response = self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {e}\nResponse: {response}")
```

---

### 3. Prompt Manager (src/prompts.py)

Load and manage prompt versions:

```python
from pathlib import Path
from typing import Dict

class PromptManager:
    """
    Manages prompt versions and loading.
    """
    
    def __init__(self, version: str = "v1.0", base_path: str = "prompts"):
        self.version = version
        self.base_path = Path(base_path)
        self.prompts_path = self.base_path / version
        
        if not self.prompts_path.exists():
            raise ValueError(f"Prompt version {version} not found at {self.prompts_path}")
    
    def load(self, prompt_name: str) -> str:
        """
        Load a prompt by name.
        
        Args:
            prompt_name: Name without extension (e.g., "evidence_extraction")
        
        Returns:
            Prompt content as string
        """
        prompt_file = self.prompts_path / f"{prompt_name}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_all(self) -> Dict[str, str]:
        """
        Load all prompts in the version directory.
        
        Returns:
            Dictionary of {prompt_name: content}
        """
        prompts = {}
        for prompt_file in self.prompts_path.glob("*.txt"):
            prompt_name = prompt_file.stem
            prompts[prompt_name] = self.load(prompt_name)
        return prompts
    
    def get_version(self) -> str:
        """Get current prompt version."""
        return self.version
```

---

### 4. Decision Logic (src/decision_logic.py)

Rule-based hiring decision (deterministic, no AI):

```python
from src.models import DimensionScore, RoleMetadata, Decision, RoleLevel
from typing import List

class DecisionEngine:
    """
    Rule-based decision logic - NO AI.
    Deterministic, explainable, consistent.
    """
    
    # Role thresholds
    THRESHOLDS = {
        RoleLevel.JUNIOR: 3.0,
        RoleLevel.INTERMEDIATE: 3.5,
        RoleLevel.SENIOR: 4.0,
        RoleLevel.LEAD: 4.2
    }
    
    # Dimension weights by role level
    WEIGHTS = {
        RoleLevel.JUNIOR: {
            "Fundamentals": 0.35,
            "Coding": 0.30,
            "Communication": 0.15,
            "Learning": 0.15,
            "Practical": 0.05
        },
        RoleLevel.INTERMEDIATE: {
            "Fundamentals": 0.25,
            "Coding": 0.25,
            "Architecture": 0.15,
            "Practical": 0.20,
            "Ownership": 0.10,
            "Communication": 0.05
        },
        RoleLevel.SENIOR: {
            "Architecture": 0.25,
            "Fundamentals": 0.20,
            "Coding": 0.20,
            "Ownership": 0.15,
            "Practical": 0.15,
            "Communication": 0.05
        },
        RoleLevel.LEAD: {
            "Architecture": 0.30,
            "Ownership": 0.25,
            "Fundamentals": 0.15,
            "Practical": 0.15,
            "Communication": 0.10,
            "Coding": 0.05
        }
    }
    
    def make_decision(
        self, 
        scores: List[DimensionScore], 
        role: RoleMetadata
    ) -> Decision:
        """
        Calculate weighted score and make hiring recommendation.
        
        Args:
            scores: List of dimension scores
            role: Role metadata
        
        Returns:
            Decision object
        """
        # Get threshold and weights for role level
        threshold = self.THRESHOLDS[role.role_level]
        weights = self.WEIGHTS[role.role_level]
        
        # Calculate weighted score
        weighted_score = 0.0
        score_dict = {s.name: s.score for s in scores}
        
        for dimension, weight in weights.items():
            if dimension in score_dict:
                weighted_score += score_dict[dimension] * weight
        
        # Check critical red flags
        red_flag, red_flag_reason = self._check_red_flags(scores, role)
        if red_flag:
            return Decision(
                recommendation="Reject",
                reasoning=f"Critical red flag: {red_flag_reason}",
                weighted_score=weighted_score,
                threshold=threshold
            )
        
        # Apply threshold logic
        if weighted_score >= threshold + 0.5:
            recommendation = "Strong Hire"
            reasoning = f"Score {weighted_score:.2f} significantly exceeds threshold {threshold}"
        elif weighted_score >= threshold:
            recommendation = "Hire"
            reasoning = f"Score {weighted_score:.2f} meets threshold {threshold}"
        elif weighted_score >= threshold - 0.3 and self._has_compensating_strengths(scores):
            recommendation = "Borderline"
            reasoning = f"Score {weighted_score:.2f} slightly below threshold {threshold} but has compensating strengths"
        else:
            recommendation = "Reject"
            reasoning = f"Score {weighted_score:.2f} below threshold {threshold}"
        
        return Decision(
            recommendation=recommendation,
            reasoning=reasoning,
            weighted_score=round(weighted_score, 2),
            threshold=threshold
        )
    
    def _check_red_flags(self, scores: List[DimensionScore], role: RoleMetadata) -> tuple[bool, str]:
        """
        Check for critical red flags that result in automatic rejection.
        
        Returns:
            (is_red_flag, reason)
        """
        score_dict = {s.name: s.score for s in scores}
        
        # Red flag: Fundamentals < 2.0 for roles requiring 3+ years
        if role.role_level in [RoleLevel.INTERMEDIATE, RoleLevel.SENIOR, RoleLevel.LEAD]:
            if score_dict.get("Fundamentals", 0) < 2.0:
                return True, "Fundamentals score below minimum for experience level"
        
        # Red flag: Coding < 2.0 for any engineering role
        if score_dict.get("Coding", 0) < 2.0:
            return True, "Coding score below minimum for engineering role"
        
        # Red flag: Ownership < 2.0 for senior+ roles
        if role.role_level in [RoleLevel.SENIOR, RoleLevel.LEAD]:
            if score_dict.get("Ownership", 0) < 2.0:
                return True, "Ownership score below minimum for senior role"
        
        return False, ""
    
    def _has_compensating_strengths(self, scores: List[DimensionScore]) -> bool:
        """
        Check if candidate has exceptional strengths in key areas.
        """
        score_dict = {s.name: s.score for s in scores}
        
        # At least 2 dimensions >= 4.5
        strong_dimensions = [s for s in scores if s.score >= 4.5]
        return len(strong_dimensions) >= 2
```

---

### 5. Core Evaluation Engine (src/engine.py)

Main orchestration logic:

```python
from src.ai_client import AIClient
from src.prompts import PromptManager
from src.decision_logic import DecisionEngine
from src.models import *
import json
import time
from datetime import datetime
import uuid

class InterviewEvaluationEngine:
    """
    Core evaluation engine - orchestrates AI calls and decision logic.
    """
    
    def __init__(
        self, 
        ai_client: AIClient, 
        prompt_manager: PromptManager,
        decision_engine: DecisionEngine
    ):
        self.ai = ai_client
        self.prompts = prompt_manager
        self.decision_engine = decision_engine
    
    def evaluate(
        self, 
        transcript: str,
        role: RoleMetadata,
        interview: InterviewMetadata,
        transcript_file: str = "",
        include_interviewer_analysis: bool = False
    ) -> EvaluationResult:
        """
        Main evaluation method.
        
        Args:
            transcript: Interview transcript text
            role: Role metadata
            interview: Interview metadata
            transcript_file: Original transcript filename
            include_interviewer_analysis: Whether to analyze interviewer (optional)
        
        Returns:
            Complete evaluation result
        """
        start_time = time.time()
        evaluation_id = str(uuid.uuid4())[:8]
        
        print(f"\n[{evaluation_id}] Starting evaluation...")
        
        # Step 1: Extract Evidence (AI Call #1)
        print(f"[{evaluation_id}] Step 1/5: Extracting evidence...")
        evidence = self._extract_evidence(transcript, role)
        
        # Step 2: Score Dimensions (AI Call #2)
        print(f"[{evaluation_id}] Step 2/5: Scoring dimensions...")
        scores = self._score_dimensions(evidence, role)
        
        # Step 3: Analyze Interviewer (AI Call #3) - Optional
        interviewer_analysis = None
        if include_interviewer_analysis:
            print(f"[{evaluation_id}] Step 3/5: Analyzing interviewer...")
            interviewer_analysis = self._analyze_interviewer(transcript, interview)
        else:
            print(f"[{evaluation_id}] Step 3/5: Skipping interviewer analysis")
        
        # Step 4: Make Decision (Rule-based)
        print(f"[{evaluation_id}] Step 4/5: Making hiring decision...")
        decision = self.decision_engine.make_decision(scores, role)
        
        # Step 5: Generate Outputs (AI Call #4a and #4b)
        print(f"[{evaluation_id}] Step 5/5: Generating feedback...")
        feedback_naresh = self._generate_naresh_feedback(
            scores, evidence, decision, role, interview
        )
        feedback_quick = self._generate_quick_summary(
            scores, decision, role, interview
        )
        
        processing_time = time.time() - start_time
        print(f"[{evaluation_id}] Completed in {processing_time:.1f}s")
        
        return EvaluationResult(
            evaluation_id=evaluation_id,
            prompt_versions={
                "evidence_extraction": self.prompts.get_version(),
                "dimension_scoring": self.prompts.get_version(),
                "output_naresh": self.prompts.get_version(),
                "output_quick": self.prompts.get_version()
            },
            transcript_file=transcript_file,
            role=role,
            interview=interview,
            evidence=evidence,
            scores=scores,
            interviewer_analysis=interviewer_analysis,
            decision=decision,
            feedback_naresh=feedback_naresh,
            feedback_quick=feedback_quick,
            processing_time_seconds=round(processing_time, 2),
            timestamp=datetime.utcnow().isoformat()
        )
    
    def _extract_evidence(self, transcript: str, role: RoleMetadata) -> Evidence:
        """
        AI Call #1: Extract evidence from transcript.
        """
        system_prompt = self.prompts.load("evidence_extraction")
        
        user_prompt = f"""
# Interview Transcript
{transcript}

# Role Requirements
{role.model_dump_json(indent=2)}

Extract evidence as specified in the system instructions.
"""
        
        response = self.ai.complete_json(system_prompt, user_prompt, temperature=0.1)
        return Evidence(**response)
    
    def _score_dimensions(self, evidence: Evidence, role: RoleMetadata) -> List[DimensionScore]:
        """
        AI Call #2: Score candidate across 7 dimensions.
        """
        system_prompt = self.prompts.load("dimension_scoring")
        
        user_prompt = f"""
# Evidence
{evidence.model_dump_json(indent=2)}

# Role Requirements
{role.model_dump_json(indent=2)}

Score the candidate on all 7 dimensions as specified in the system instructions.
"""
        
        response = self.ai.complete_json(system_prompt, user_prompt, temperature=0.2)
        return [DimensionScore(**dim) for dim in response["dimensions"]]
    
    def _analyze_interviewer(
        self, 
        transcript: str, 
        interview: InterviewMetadata
    ) -> InterviewerAnalysis:
        """
        AI Call #3: Analyze interviewer quality (optional).
        """
        # Note: This is placeholder for Phase 0 - can be simplified or skipped
        # For full implementation, add interviewer_analysis prompt
        return None
    
    def _generate_naresh_feedback(
        self,
        scores: List[DimensionScore],
        evidence: Evidence,
        decision: Decision,
        role: RoleMetadata,
        interview: InterviewMetadata
    ) -> str:
        """
        AI Call #4a: Generate Naresh-style feedback.
        """
        system_prompt = self.prompts.load("output_naresh")
        
        user_prompt = f"""
# Scores
{json.dumps([s.model_dump() for s in scores], indent=2)}

# Evidence
{evidence.model_dump_json(indent=2)}

# Decision
{decision.model_dump_json(indent=2)}

# Role
{role.model_dump_json(indent=2)}

# Interview Metadata
Interviewer: {interview.interviewer}
Duration: {interview.duration_minutes} minutes
Round: {interview.round_type}

Generate Naresh-style feedback as specified in the system instructions.
"""
        
        return self.ai.complete(system_prompt, user_prompt, temperature=0.3)
    
    def _generate_quick_summary(
        self,
        scores: List[DimensionScore],
        decision: Decision,
        role: RoleMetadata,
        interview: InterviewMetadata
    ) -> str:
        """
        AI Call #4b: Generate quick summary.
        """
        system_prompt = self.prompts.load("output_quick_summary")
        
        user_prompt = f"""
# Scores
{json.dumps([s.model_dump() for s in scores], indent=2)}

# Decision
{decision.model_dump_json(indent=2)}

# Role
{role.model_dump_json(indent=2)}

Generate a quick recruiter-friendly summary (150-250 words) as specified in the system instructions.
"""
        
        return self.ai.complete(system_prompt, user_prompt, temperature=0.3)
```

---

### 6. Main Execution Script (scripts/run_evaluation.py)

Command-line interface:

```python
#!/usr/bin/env python3
"""
Main script to run interview evaluation.

Usage:
    python scripts/run_evaluation.py --transcript tests/transcripts/transcript_1.txt --version v1.0
"""

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

from src.engine import InterviewEvaluationEngine
from src.ai_client import AIClient
from src.prompts import PromptManager
from src.decision_logic import DecisionEngine
from src.models import RoleMetadata, InterviewMetadata

# Load environment variables
load_dotenv()

console = Console()

def load_transcript(transcript_path: Path) -> str:
    """Load transcript from file."""
    with open(transcript_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_metadata(metadata_path: Path) -> tuple[RoleMetadata, InterviewMetadata]:
    """Load role and interview metadata from JSON file."""
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    role = RoleMetadata(**data["role"])
    interview = InterviewMetadata(**data["interview"])
    return role, interview

def save_result(result, output_path: Path):
    """Save evaluation result to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save full JSON result
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(result.model_dump(), f, indent=2)
    
    # Save Naresh-style feedback as markdown
    md_path = output_path.with_suffix('.md')
    with open(md_path, 'w') as f:
        f.write(result.feedback_naresh)
    
    console.print(f"\n✅ Results saved:")
    console.print(f"  - JSON: {json_path}")
    console.print(f"  - Markdown: {md_path}")

def main():
    parser = argparse.ArgumentParser(description="Run interview evaluation")
    parser.add_argument(
        "--transcript",
        type=Path,
        required=True,
        help="Path to transcript file"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0",
        help="Prompt version to use (default: v1.0)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: tests/results/{version}/evaluation_{id})"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.transcript.exists():
        console.print(f"[red]Error: Transcript file not found: {args.transcript}[/red]")
        return 1
    
    # Load transcript
    console.print(f"\n📄 Loading transcript: {args.transcript}")
    transcript = load_transcript(args.transcript)
    
    # Load metadata (assumes {transcript_name}_metadata.json exists)
    metadata_path = args.transcript.parent / f"{args.transcript.stem}_metadata.json"
    if not metadata_path.exists():
        console.print(f"[red]Error: Metadata file not found: {metadata_path}[/red]")
        console.print("[yellow]Expected format: {transcript_name}_metadata.json[/yellow]")
        return 1
    
    role, interview = load_metadata(metadata_path)
    
    # Initialize components
    console.print(f"\n🤖 Initializing evaluation engine (prompts: {args.version})...")
    ai_client = AIClient()
    prompt_manager = PromptManager(version=args.version)
    decision_engine = DecisionEngine()
    engine = InterviewEvaluationEngine(ai_client, prompt_manager, decision_engine)
    
    # Run evaluation
    console.print(f"\n⚙️  Processing evaluation...\n")
    result = engine.evaluate(
        transcript=transcript,
        role=role,
        interview=interview,
        transcript_file=str(args.transcript),
        include_interviewer_analysis=False  # Phase 0: skip interviewer analysis
    )
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_dir = Path("tests/results") / args.version
        output_path = output_dir / f"evaluation_{result.evaluation_id}"
    
    # Save results
    save_result(result, output_path)
    
    # Display summary
    console.print("\n" + "="*70)
    console.print(f"📊 EVALUATION SUMMARY")
    console.print("="*70 + "\n")
    console.print(f"Evaluation ID: {result.evaluation_id}")
    console.print(f"Role: {role.role_title} ({role.role_level.value})")
    console.print(f"Decision: [bold]{result.decision.recommendation}[/bold]")
    console.print(f"Weighted Score: {result.decision.weighted_score:.2f} (threshold: {result.decision.threshold})")
    console.print(f"Processing Time: {result.processing_time_seconds}s")
    
    console.print("\n📈 Dimension Scores:")
    for score in result.scores:
        console.print(f"  • {score.name}: {score.score:.1f}/5")
    
    console.print("\n📝 Quick Summary:")
    console.print(Markdown(result.feedback_quick))
    
    console.print("\n💬 Full Feedback (Naresh-style):")
    console.print(Markdown(result.feedback_naresh))
    
    return 0

if __name__ == "__main__":
    exit(main())
```

---

## Prompt Templates

### Evidence Extraction Prompt (prompts/v1.0/evidence_extraction.txt)

```
You are analyzing a technical interview transcript to extract evidence for candidate evaluation.

Your task is to extract structured evidence that will be used for scoring. Be precise, quote exact text, and distinguish between depth levels.

# Instructions

1. **Technical Statements**
   - Extract every technical claim the candidate made
   - Quote exact text from transcript
   - Identify the topic (e.g., "HashMap internals", "Microservices design", "Python decorators")
   - Classify depth:
     * "superficial": Mentions concept without explanation
     * "detailed": Explains how it works with some specifics
     * "expert": Deep explanation with tradeoffs, edge cases, or implementation details

2. **Ownership vs Exposure Detection**
   - Look for ownership language: "I designed", "I chose", "I evaluated", "I decided"
   - vs exposure language: "we used", "the team decided", "it was set up", "the company had"
   - Quote exact phrases that indicate ownership or exposure
   - Provide context for each signal

3. **Problem-Solving Approach**
   - How did the candidate approach coding problems?
   - Did they clarify requirements first?
   - What was their debugging approach?
   - Did they handle edge cases?

4. **Buzzword Flagging**
   - Flag technical terms used without explanation
   - Identify when interviewer probed deeper and candidate couldn't elaborate
   - Examples: saying "microservices" without explaining why/how, mentioning "scalable" without specifics

# Output Format

Return a JSON object with this structure:

{
  "technical_statements": [
    {
      "topic": "HashMap",
      "quote": "HashMap uses chaining for collision resolution, and in Java 8+ it converts to a tree structure when a bucket gets too large",
      "depth": "detailed",
      "evidence_type": "strength",
      "context": "Explained fundamentals clearly with version-specific details"
    }
  ],
  "ownership_signals": [
    {
      "type": "ownership",
      "quote": "I designed the event-driven architecture using Kafka",
      "context": "Architecture discussion - clear ownership of design decision"
    },
    {
      "type": "exposure",
      "quote": "We used microservices at my company",
      "context": "No details on why, how, or personal contribution"
    }
  ],
  "problem_solving_approach": {
    "clarifies_requirements": true,
    "systematic_debugging": false,
    "handles_edge_cases": true,
    "notes": "Asked clarifying questions before coding. Missed some edge cases initially but caught them when prompted."
  },
  "buzzwords_flagged": ["microservices", "scalable", "cloud-native"]
}

# Critical Rules

- Quote EXACT text from the transcript - do not paraphrase in quotes
- Be honest about depth - "superficial" is valid if candidate couldn't explain deeper
- Distinguish ownership from exposure carefully - this matters for seniority
- Flag buzzwords even if candidate seems confident - lack of detail = buzzword
```

---

### Dimension Scoring Prompt (prompts/v1.0/dimension_scoring.txt)

```
You are an expert technical interviewer scoring a candidate across 7 dimensions.

You have been provided with extracted evidence from the interview. Your task is to assign precise scores (0-5 scale) to each dimension based on the evidence and the scoring rubrics.

# Dimensions to Score

1. **Fundamentals** (0-5)
2. **Coding & Problem Solving** (0-5)
3. **Architecture & System Design** (0-5)
4. **Ownership & Initiative** (0-5)
5. **Communication** (0-5)
6. **Practical Experience** (0-5)
7. **Learning Ability** (0-5)

# Scoring Rubrics

## Fundamentals (0-5)

- **0-1**: Cannot explain basic concepts, incorrect definitions (e.g., confuses async/sync, wrong SOLID explanations)
- **2**: Surface-level understanding, memorized definitions without reasoning (e.g., explains "what" but not "why")
- **3**: Solid understanding, can explain with examples (e.g., clear explanations, provides examples, connects concepts)
- **4**: Strong fundamentals, explains tradeoffs and nuances (e.g., discusses ArrayList vs LinkedHashMap tradeoffs, explains "why")
- **5**: Expert-level, deep understanding, can teach concepts (e.g., explains JVM internals, memory models, advanced patterns)

## Coding & Problem Solving (0-5)

- **0-1**: Cannot solve basic problems, needs heavy guidance (e.g., stuck on fizzbuzz, no systematic approach)
- **2**: Solves with significant help, basic approach (e.g., brute force only, interviewer-guided)
- **3**: Solves independently with decent approach (e.g., clear approach, working code, handles basic edge cases)
- **4**: Strong problem solving, optimal solutions, good debugging (e.g., optimal solution, discusses alternatives, systematic debugging)
- **5**: Exceptional problem solving, multiple approaches, teaches concepts (e.g., elegant solution, explains tradeoffs, anticipates edge cases)

## Architecture & System Design (0-5)

- **0-1**: No system design understanding, cannot decompose (e.g., cannot break problem into services, no distributed knowledge)
- **2**: Basic understanding, buzzword usage without depth (e.g., "we use microservices" but cannot explain why)
- **3**: Solid understanding, can design services with prompting (e.g., identifies services, explains communication, basic scalability awareness)
- **4**: Strong architecture skills, evaluates tradeoffs independently (e.g., discusses CAP tradeoffs, considers failure modes, evaluates async vs sync)
- **5**: Expert-level architecture, considers org impact, capacity planning (e.g., multi-system design, evaluates consistency models, operational considerations)

## Ownership & Initiative (0-5)

- **0-1**: Task-level only, no ownership (e.g., "I was told to...", no initiative examples)
- **2**: Completes assigned work well (e.g., "I implemented the feature as specified", minimal proactivity)
- **3**: Feature-level ownership, some initiative (e.g., "I suggested improvements", "I refactored this module")
- **4**: System-level ownership, drives decisions (e.g., "I led the migration", "I proposed the architecture", mentors juniors)
- **5**: Org-level impact, strategic thinking (e.g., "I designed the platform", "I established the pattern", influences standards)

## Communication (0-5)

- **0-1**: Cannot explain technical concepts clearly (e.g., rambling, incoherent, doesn't answer questions)
- **2**: Basic communication, struggles with structure (e.g., jumps around, needs rephrasing, misses question intent)
- **3**: Clear communication, structured responses (e.g., structured answers, stays on topic, clear examples)
- **4**: Strong communication, teaches effectively (e.g., crystal clear explanations, uses analogies, adapts to interviewer)
- **5**: Exceptional communication, influences thinking (e.g., teaches concepts naturally, exceptional clarity, anticipates confusion)

## Practical Experience (0-5)

- **0-1**: No practical experience, theoretical only (e.g., cannot describe implementation details, no production stories)
- **2**: Limited exposure, worked on features (e.g., describes tasks but not system context, limited production debugging)
- **3**: Solid practical experience, understands production (e.g., can describe debugging sessions, monitoring usage, incident handling)
- **4**: Strong practical depth, production ownership (e.g., production war stories, performance tuning examples, operational maturity)
- **5**: Expert-level practical experience, teaches others (e.g., cross-system production experience, architectural evolution stories, post-mortems)

## Learning Ability (0-5)

- **0-1**: Resistant to learning, bluffs through gaps (e.g., fakes knowledge, defensive about gaps)
- **2**: Minimal learning examples, narrow focus (e.g., rarely learns outside work, stuck in comfort zone)
- **3**: Learns when needed, steady growth (e.g., can describe learning journeys, adapts to new tech)
- **4**: Active learner, explores beyond requirements (e.g., stays current, explores tangential domains, shares learnings)
- **5**: Exceptional learning ability, thought leader (e.g., anticipates tech trends, deep cross-domain knowledge, teaches industry)

# Output Format

Return a JSON object with this structure:

{
  "dimensions": [
    {
      "name": "Fundamentals",
      "score": 3.5,
      "reasoning": "Solid understanding of HashMap internals (chaining, tree conversion). Explained OOP concepts clearly. However, struggled with distributed transaction patterns when probed deeper. Knows basics well but gaps in advanced topics.",
      "evidence": [
        {
          "type": "strength",
          "quote": "HashMap uses chaining for collision resolution...",
          "topic": "HashMap"
        },
        {
          "type": "weakness",
          "quote": "Could not explain 2PC vs Saga patterns beyond basic ACID",
          "topic": "Distributed transactions"
        }
      ]
    },
    {
      "name": "Coding",
      "score": 3.0,
      "reasoning": "...",
      "evidence": [...]
    }
    // ... continue for all 7 dimensions
  ]
}

# Critical Rules

- Score MUST be based on evidence provided - do not invent
- Use the full 0-5 range with decimals (e.g., 3.5, 4.2)
- Provide specific reasoning that references the evidence
- Quote supporting evidence for both strengths and weaknesses
- Be precise: 3.5 vs 4.0 matters - use the rubric carefully
- If evidence is insufficient for a dimension, note it in reasoning and score conservatively
```

---

### Naresh-Style Output Prompt (prompts/v1.0/output_naresh.txt)

```
You are generating a "Naresh-style" interview feedback - direct, grounded, evidence-backed, with no fluff.

# Style Characteristics

- **Direct**: Get to the point immediately
- **Grounded**: Every claim backed by specific evidence
- **Practical**: Focus on hiring decision, not academic analysis
- **No fluff**: No generic HR language ("team player", "passionate", "motivated")
- **Concise**: 300-500 words total
- **Honest**: Call out weaknesses directly but professionally

# Structure

```
# [Role Title] Interview - Candidate [Name]

**Status:** [Strong Hire | Hire | Borderline | Reject]

## Remarks

[Opening: Candidate summary - experience, current role]

[Competency analysis for each key dimension - direct statements with evidence]
- Fundamentals: [Score/5] - [Specific finding with quote/example]
- Coding: [Score/5] - [Specific finding with quote/example]
- Architecture: [Score/5] - [Specific finding with quote/example]

[Key concerns or strengths - specific points with examples]

[Role calibration - does level match performance?]

## Interview Reliability Note

**Reliability:** [High | Medium | Low]

[Brief assessment of interview quality]

## Overall Take

[2-3 sentences: Bottom line assessment, practical hiring perspective]

## One-Line Summary

[Single sentence capturing essence]
```

# Example (for reference)

```
# Senior Backend Developer Interview - Candidate X

**Status:** Reject

## Remarks

Candidate has 8 years experience, currently Backend Developer at a product company. Resume highlights microservices migration, payment processing, and team lead role.

Fundamentals are below senior bar (3.5/5). Could not explain distributed transaction handling beyond ACID basics. When asked about 2PC vs Saga patterns, response was "we used transactions in the database." HashMap explanation was solid but struggled with advanced concurrency patterns.

Coding is adequate (3/5) but not senior level. Solved the string manipulation problem but needed hints on optimization. Initial approach was O(n²), only reached O(n) after prompting. No systematic edge case handling - missed empty string case until pointed out.

Architecture reveals exposure but not ownership (2.5/5). Mentioned "worked on microservices" but could not explain service boundaries or why microservices were chosen. When asked about event-driven architecture, said "we used Kafka" but no details on partitioning, consumer groups, or failure handling.

Calibration: Strong Intermediate, not Senior. Has good exposure to modern stack but lacks architectural depth and strong fundamentals expected at 8 years.

## Interview Reliability Note

**Reliability:** High

Interviewer did excellent job - 50 minutes, covered fundamentals + coding + architecture, multiple follow-ups on vague answers. Candidate given ample opportunity to demonstrate depth.

## Overall Take

Worked in good environment and has exposure to right tools, but hasn't developed senior-level depth. Fundamentals gap is concerning at 8 years. Would not recommend for senior role.

## One-Line Summary

Exposure present, depth missing - not senior level.
```

# Critical Rules

- Back EVERY claim with evidence (quote or specific example)
- Distinguish "awareness" from "exposure" from "ownership"
- Specify what's missing for the target role
- Be direct about weaknesses - no sugar-coating
- Keep it concise - aim for 300-500 words
- Use scores inline (X/5) for key dimensions
```

---

### Quick Summary Prompt (prompts/v1.0/output_quick_summary.txt)

```
You are generating a quick, recruiter-friendly interview summary.

# Characteristics

- **Audience**: Recruiters, hiring managers (may not be deeply technical)
- **Length**: 150-250 words
- **Tone**: Direct, actionable, jargon-free
- **Purpose**: Quick decision - proceed or pass?

# Structure

```
# Quick Interview Summary

**Candidate:** [Name]
**Role:** [Role Title]
**Recommendation:** [Hire | Borderline | Reject]

## Quick Take
[2-3 sentences on overall impression and key concern/strength]

## Key Points
- ✓ [Strength 1 with brief evidence]
- ✓ [Strength 2 with brief evidence]
- ✗ [Concern 1 with brief evidence]
- ✗ [Concern 2 with brief evidence]

## Next Steps
[Clear recommendation: Proceed to next round | Additional technical round | Pass]
```

# Example

```
# Quick Interview Summary

**Candidate:** Candidate A
**Role:** Senior Python Developer
**Recommendation:** Reject

## Quick Take
Candidate has good exposure to modern Python stack (Django, FastAPI) but fundamentals and problem-solving are below senior level. Struggled with core concepts despite 8 years stated experience.

## Key Points
- ✗ Weak fundamentals (2/5): Could not explain advanced concepts, confused async patterns
- ✗ Coding below senior level (2.5/5): Needed significant guidance, brute force approach only
- ✓ Good communication: Articulates ideas clearly, structured responses
- ✗ Architecture depth lacking: "Worked on microservices" but couldn't explain distributed patterns

## Next Steps
Do not proceed. Fundamentals gap is too significant for senior role. May be suitable for intermediate position after addressing core concepts.
```

# Critical Rules

- Use checkmarks (✓) and crosses (✗) for visual scanning
- Keep each point to one line when possible
- Be direct in "Next Steps" - clear action item
- Avoid technical jargon where possible
- Make recommendation crystal clear
```

---

## Configuration Files

### Role Definitions (config/role_definitions.json)

```json
{
  "senior_backend": {
    "role_title": "Senior Backend Developer",
    "role_level": "senior",
    "experience_years": "6-10",
    "must_have_skills": [
      {
        "skill": "Python",
        "depth": "expert",
        "importance": "critical",
        "context": "Production-scale async processing"
      },
      {
        "skill": "Django",
        "depth": "proficient",
        "importance": "critical",
        "context": "REST API development, ORM optimization"
      },
      {
        "skill": "System Design",
        "depth": "expert",
        "importance": "critical",
        "context": "Design distributed systems from scratch"
      }
    ],
    "nice_to_have": ["GraphQL", "Kubernetes", "Redis"],
    "key_responsibilities": [
      "Design backend services and APIs",
      "Lead technical architecture decisions",
      "Mentor junior and mid-level engineers",
      "Own production systems and incident response"
    ]
  }
}
```

### Scoring Rubrics (config/scoring_rubrics.json)

```json
{
  "dimensions": {
    "Fundamentals": {
      "0-1": "Cannot explain basic concepts, incorrect definitions",
      "2": "Surface-level understanding, memorized definitions",
      "3": "Solid understanding, can explain with examples",
      "4": "Strong fundamentals, explains tradeoffs",
      "5": "Expert-level, deep understanding, teaches concepts"
    },
    "Coding": {
      "0-1": "Cannot solve basic problems, needs heavy guidance",
      "2": "Solves with significant help, basic approach",
      "3": "Solves independently with decent approach",
      "4": "Strong problem solving, optimal solutions",
      "5": "Exceptional, multiple approaches, elegant code"
    }
  }
}
```

---

## Test Data Format

### Transcript (tests/transcripts/transcript_1.txt)

```
[Plain text format]

Interviewer: Can you explain how HashMap works in Java?

Candidate: HashMap uses hashing to store key-value pairs. It uses the hashCode() method to determine which bucket to place an entry in. When there's a collision, it uses chaining - a linked list of entries in the same bucket. In Java 8 and later, if a bucket gets too many entries, it converts the linked list to a balanced tree for better performance.

Interviewer: Good. Can you explain the distributed architecture at your current company?

Candidate: We use microservices. Each service handles a different part of the business logic.

Interviewer: What communication pattern do the services use?

Candidate: We use Kafka for messaging between services.

Interviewer: Why did you choose Kafka over other options?

Candidate: The team decided to use Kafka before I joined.

[... continue transcript ...]
```

### Metadata (tests/transcripts/transcript_1_metadata.json)

```json
{
  "role": {
    "role_title": "Senior Backend Developer",
    "role_level": "senior",
    "experience_years": "6-10",
    "must_have_skills": [
      {
        "skill": "Python",
        "depth": "expert",
        "importance": "critical",
        "context": "Production-scale development"
      },
      {
        "skill": "System Design",
        "depth": "expert",
        "importance": "critical",
        "context": "Distributed systems"
      }
    ],
    "nice_to_have": ["Kafka", "Redis"],
    "key_responsibilities": [
      "Design backend architecture",
      "Mentor team members"
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

### Ground Truth (tests/transcripts/transcript_1_groundtruth.json)

```json
{
  "actual_decision": "Reject",
  "actual_reasoning": "Fundamentals below senior bar, architecture knowledge is exposure not ownership",
  "critical_issues_to_catch": [
    "Could not explain distributed architecture decisions (why Kafka, what tradeoffs)",
    "No ownership language - 'team decided' vs 'I chose'",
    "Microservices explanation too shallow for senior level",
    "Good fundamentals (HashMap) but weak on distributed systems"
  ],
  "expected_recommendation": "Reject",
  "expected_calibration": "Strong Intermediate, not Senior"
}
```

---

## Documentation

### README.md

Include:
- Project overview
- Setup instructions (Python environment, OpenAI API key)
- Usage examples
- Prompt versioning workflow
- How to add test transcripts
- Interpretation of results

### Prompt Iteration Log (docs/PROMPT_ITERATION_LOG.md)

Track prompt changes and their impact:

```markdown
# Prompt Iteration Log

## v1.1 (2025-05-20)

### Changes
- **evidence_extraction.txt**: Added explicit instruction to flag buzzwords when candidate can't elaborate
- **dimension_scoring.txt**: Clarified ownership scoring - "team decided" should score lower

### Reason
Test transcript #3 missed buzzword detection (candidate said "microservices" without explaining). Test transcript #5 over-scored ownership when candidate only had exposure.

### Results
- Buzzword detection: 70% → 95% accuracy
- Ownership scoring: 3 false positives → 0

### Test Comparison
| Transcript | v1.0 Score | v1.1 Score | Ground Truth | v1.1 Better? |
|------------|------------|------------|--------------|--------------|
| #1 | 3.2 | 3.2 | Reject | Same |
| #3 | 3.8 (miss) | 3.5 (hit) | Reject | ✅ Yes |
| #5 | 4.1 (miss) | 3.7 (hit) | Reject | ✅ Yes |

### Decision
**Promote v1.1 to baseline** - clear improvement in accuracy
```

---

## Acceptance Criteria

Phase 0 is complete when:

1. ✅ **Core engine works**: Can process a transcript and generate evaluation
2. ✅ **All 4 AI calls implemented**: Evidence extraction, scoring, Naresh feedback, quick summary
3. ✅ **Decision logic works**: Rule-based scoring produces correct hire/reject recommendations
4. ✅ **Prompt versioning works**: Can switch between prompt versions and compare results
5. ✅ **Test suite passes**: 5-10 transcripts evaluated with ≥90% accuracy
6. ✅ **Output quality validated**: Hiring managers confirm feedback is useful and accurate
7. ✅ **Documentation complete**: README, prompt iteration log, test data format documented
8. ✅ **No hallucinations**: Zero cases of AI fabricating major issues not in transcript
9. ✅ **Evidence-backed**: Every claim in feedback has supporting quote from transcript
10. ✅ **Ready for Gate 1**: Can make go/no-go decision on proceeding to Phase 1

---

## Success Metrics

Track these for each test transcript:

```json
{
  "transcript_id": "transcript_1",
  "ground_truth": {
    "actual_decision": "Reject",
    "critical_issues": [
      "Weak distributed systems understanding",
      "Exposure not ownership"
    ]
  },
  "ai_evaluation": {
    "decision": "Reject",
    "issues_caught": [
      "Architecture knowledge is exposure (2.5/5)",
      "No ownership language detected"
    ]
  },
  "accuracy_analysis": {
    "decision_match": true,
    "issues_caught": 2,
    "issues_total": 2,
    "false_positives": 0,
    "accuracy_score": 100
  }
}
```

**Overall Phase 0 Success:**
- Average accuracy ≥ 90% across all test transcripts
- Zero critical false positives (fabricated major issues)
- Hiring manager validation: "This is useful, I would use this"

---

## Next Steps After Phase 0

**If Phase 0 succeeds (≥90% accuracy):**

1. Document lessons learned
2. Finalize prompt versions for production
3. Plan Phase 1 (MVP) architecture
4. Budget approval for Azure infrastructure
5. Begin Phase 1 development

**If Phase 0 needs iteration (<90% accuracy):**

1. Analyze failure patterns
2. Refine prompts based on specific issues
3. Re-test with improved prompts
4. Repeat until 90% threshold met

---

## Implementation Notes for Claude Code

- **Use type hints throughout** - Python 3.10+ with Pydantic models
- **Error handling** - Graceful failures with informative messages
- **Logging** - Use `rich` for readable console output
- **Testing** - Include sample test data in `tests/transcripts/`
- **Configuration** - Use `.env` for API keys, JSON for role definitions
- **Modularity** - Each component (AI client, prompts, engine) should be independently testable
- **Documentation** - Docstrings for all classes and methods

---

## Questions or Clarifications Needed

Before starting implementation, please confirm:

1. Should interviewer analysis (AI Call #3) be included in Phase 0, or added later?
   - **Recommendation**: Skip for Phase 0, add in Phase 1 (focus on core evaluation first)

2. Should we support JD integration in Phase 0?
   - **Recommendation**: No, test core evaluation without JD first

3. Any specific formatting requirements for output files?
   - Current plan: JSON for structured data, Markdown for human-readable feedback

4. Should we include a comparison tool for evaluating prompt versions side-by-side?
   - **Recommendation**: Yes, include `scripts/compare_versions.py` for easy A/B comparison

---

## Estimated Completion

- **Setup & scaffolding**: 1 day
- **Core engine implementation**: 2-3 days
- **Prompt writing & refinement**: 2-3 days
- **Testing & iteration**: 3-5 days
- **Documentation**: 1 day

**Total: 2-3 weeks** with 1 developer working full-time, or 3-4 weeks part-time.
