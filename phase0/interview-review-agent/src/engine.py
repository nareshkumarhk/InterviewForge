from src.ai_client import AIClient, RateLimitExhausted
from src.transcript_compressor import compress_rules, compress_ai, _AI_CHUNK_TOKENS
from src.prompts import PromptManager
from src.decision_logic import DecisionEngine
from src.models import (
    Evidence,
    EvaluationResult,
    RoleMetadata,
    InterviewMetadata,
    InterviewerAnalysis,
    DimensionScore,
    Decision,
)
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
import uuid


# Max tokens to send as transcript content.
# Set conservatively below the TPM limit to leave room for system prompts + metadata.
# Override via TRANSCRIPT_TOKEN_BUDGET env var.
_DEFAULT_TOKEN_BUDGET = 26_000


class InterviewEvaluationEngine:
    """Core evaluation engine - orchestrates AI calls and decision logic."""

    def __init__(
        self,
        ai_client: AIClient,
        prompt_manager: PromptManager,
        decision_engine: DecisionEngine,
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
        include_interviewer_analysis: bool = False,
        partial_output_dir: Optional[Path] = None,
    ) -> EvaluationResult:
        """
        Main evaluation method.

        On RateLimitExhausted, saves a partial result JSON (scores + decision if available)
        and re-raises so the caller can display retry information.
        """
        start_time = time.time()
        evaluation_id = str(uuid.uuid4())[:8]

        # Track partial state so we can save on rate-limit failure
        evidence: Optional[Evidence] = None
        scores: Optional[List[DimensionScore]] = None
        decision: Optional[Decision] = None

        # Reserve ~2,000 tokens for system prompt + role metadata overhead
        transcript, estimated_tokens = self._fit_transcript(transcript, evaluation_id)

        try:
            print(f"\n[{evaluation_id}] Starting evaluation...")
            print(f"[{evaluation_id}] Transcript: ~{estimated_tokens:,} tokens")
            print(f"[{evaluation_id}] Step 1/5: Extracting evidence...")
            evidence = self._extract_evidence(transcript, role)
            self._print_rate_limit(evaluation_id, "evidence extraction")

            print(f"[{evaluation_id}] Step 2/5: Scoring dimensions...")
            scores = self._score_dimensions(evidence, role)
            self._print_rate_limit(evaluation_id, "dimension scoring")

            interviewer_analysis = None
            if include_interviewer_analysis:
                print(f"[{evaluation_id}] Step 3/5: Analyzing interviewer...")
                interviewer_analysis = self._analyze_interviewer(transcript, interview)
                self._print_rate_limit(evaluation_id, "interviewer analysis")
            else:
                print(f"[{evaluation_id}] Step 3/5: Skipping interviewer analysis")

            print(f"[{evaluation_id}] Step 4/5: Making hiring decision...")
            decision = self.decision_engine.make_decision(scores, role)

            print(f"[{evaluation_id}] Step 5/5: Generating feedback...")
            feedback_naresh = self._generate_naresh_feedback(scores, evidence, decision, role, interview)
            feedback_quick = self._generate_quick_summary(scores, decision, role)

            self._print_rate_limit(evaluation_id, "feedback generation")

        except RateLimitExhausted:
            partial_path = self._save_partial(
                evaluation_id, evidence, scores, decision, role, interview,
                transcript_file, partial_output_dir,
            )
            print(f"[{evaluation_id}] Rate limit exhausted. Partial results saved to: {partial_path}")
            raise

        processing_time = time.time() - start_time
        print(f"[{evaluation_id}] Completed in {processing_time:.1f}s")

        return EvaluationResult(
            evaluation_id=evaluation_id,
            prompt_versions={
                "evidence_extraction": self.prompts.get_version(),
                "dimension_scoring": self.prompts.get_version(),
                "output_naresh": self.prompts.get_version(),
                "output_quick": self.prompts.get_version(),
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
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _save_partial(
        self,
        evaluation_id: str,
        evidence: Optional[Evidence],
        scores: Optional[List[DimensionScore]],
        decision: Optional[Decision],
        role: RoleMetadata,
        interview: InterviewMetadata,
        transcript_file: str,
        output_dir: Optional[Path],
    ) -> Path:
        """Save whatever was computed before the rate limit hit."""
        output_dir = output_dir or Path("tests/results/partial")
        output_dir.mkdir(parents=True, exist_ok=True)

        completed_steps = []
        if evidence:
            completed_steps.append("evidence_extraction")
        if scores:
            completed_steps.append("dimension_scoring")
        if decision:
            completed_steps.append("decision")

        partial = {
            "status": "partial",
            "evaluation_id": evaluation_id,
            "failed_at": _next_step(completed_steps),
            "completed_steps": completed_steps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transcript_file": transcript_file,
            "role": role.model_dump(),
            "interview": interview.model_dump(),
            "evidence": evidence.model_dump() if evidence else None,
            "scores": [s.model_dump() for s in scores] if scores else None,
            "decision": decision.model_dump() if decision else None,
        }

        path = output_dir / f"partial_{evaluation_id}.json"
        with open(path, "w") as f:
            json.dump(partial, f, indent=2)
        return path

    def _extract_evidence(self, transcript: str, role: RoleMetadata) -> Evidence:
        """AI Call #1: Extract evidence from transcript."""
        system_prompt = self.prompts.load("evidence_extraction")
        user_prompt = f"""# Interview Transcript
{transcript}

# Role Requirements
{role.model_dump_json(indent=2)}

Extract evidence as specified in the system instructions.
"""
        response = self.ai.complete_json(system_prompt, user_prompt, temperature=0.1)
        return Evidence(**response)

    def _score_dimensions(self, evidence: Evidence, role: RoleMetadata) -> List[DimensionScore]:
        """AI Call #2: Score candidate across 7 dimensions."""
        system_prompt = self.prompts.load("dimension_scoring")
        user_prompt = f"""# Evidence
{evidence.model_dump_json(indent=2)}

# Role Requirements
{role.model_dump_json(indent=2)}

Score the candidate on all 7 dimensions as specified in the system instructions.
"""
        response = self.ai.complete_json(system_prompt, user_prompt, temperature=0.2)
        return [DimensionScore(**dim) for dim in response["dimensions"]]

    def _analyze_interviewer(
        self, _transcript: str, _interview: InterviewMetadata
    ) -> Optional[InterviewerAnalysis]:
        """AI Call #3: Analyze interviewer quality (optional, Phase 1+)."""
        return None

    def _generate_naresh_feedback(
        self,
        scores: List[DimensionScore],
        evidence: Evidence,
        decision: Decision,
        role: RoleMetadata,
        interview: InterviewMetadata,
    ) -> str:
        """AI Call #4a: Generate Naresh-style feedback."""
        system_prompt = self.prompts.load("output_naresh")
        user_prompt = f"""# Scores
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
    ) -> str:
        """AI Call #4b: Generate quick recruiter-friendly summary."""
        system_prompt = self.prompts.load("output_quick_summary")
        user_prompt = f"""# Scores
{json.dumps([s.model_dump() for s in scores], indent=2)}

# Decision
{decision.model_dump_json(indent=2)}

# Role
{role.model_dump_json(indent=2)}

Generate a quick recruiter-friendly summary (150-250 words) as specified in the system instructions.
"""
        return self.ai.complete(system_prompt, user_prompt, temperature=0.3)

    def _fit_transcript(self, transcript: str, evaluation_id: str) -> Tuple[str, int]:
        """
        Compress transcript to fit within the token budget.

        Stage 1 (always): rule-based cleanup — strips Teams/Zoom timestamp noise,
                          repeated speaker labels, filler words, ack-only lines.
        Stage 2 (if needed): AI compression pass — preserves all technical content,
                             further condenses verbose phrasing.
        """
        budget = int(os.getenv("TRANSCRIPT_TOKEN_BUDGET", _DEFAULT_TOKEN_BUDGET))

        # Stage 1: rule-based (free)
        compressed, original_tokens, tokens_after_rules = compress_rules(transcript)
        saved = original_tokens - tokens_after_rules
        print(
            f"[{evaluation_id}] Compression stage 1 (rules): "
            f"{original_tokens:,} → {tokens_after_rules:,} tokens  (saved {saved:,})"
        )

        if tokens_after_rules <= budget:
            return compressed, tokens_after_rules

        # Stage 2: AI compression (only if still over budget)
        n_chunks = max(1, tokens_after_rules // _AI_CHUNK_TOKENS)
        if n_chunks > 1:
            chunk_info = f"{n_chunks} chunks"
        else:
            chunk_info = "single pass"
        print(
            f"[{evaluation_id}] Still over budget ({tokens_after_rules:,} > {budget:,}). "
            f"Running AI compression pass ({chunk_info})..."
        )
        compressed, tokens_after_ai = compress_ai(compressed, self.ai, budget)
        saved_ai = tokens_after_rules - tokens_after_ai
        print(
            f"[{evaluation_id}] Compression stage 2 (AI): "
            f"{tokens_after_rules:,} → {tokens_after_ai:,} tokens  (saved {saved_ai:,})"
        )
        return compressed, tokens_after_ai

    def _print_rate_limit(self, evaluation_id: str, step: str) -> None:
        rl = self.ai.last_rate_limit
        if rl and rl.remaining_requests is not None:
            print(
                f"[{evaluation_id}]   rate-limit after {step}: "
                f"requests {rl.remaining_requests}/{rl.limit_requests}"
                f" | tokens {rl.remaining_tokens}/{rl.limit_tokens}"
                f" | resets in {rl.reset_requests}"
            )


def _next_step(completed: list) -> str:
    pipeline = ["evidence_extraction", "dimension_scoring", "decision", "feedback_generation"]
    for step in pipeline:
        if step not in completed:
            return step
    return "unknown"
