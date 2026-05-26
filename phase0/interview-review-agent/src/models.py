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
    experience_years: str
    must_have_skills: List[Skill]
    nice_to_have: Optional[List[str]] = []
    key_responsibilities: List[str]


class InterviewMetadata(BaseModel):
    round_type: str
    interviewer: str
    duration_minutes: int
    date: Optional[str] = None


class EvidenceItem(BaseModel):
    topic: str
    quote: str
    depth: Literal["superficial", "detailed", "expert"]
    evidence_type: Literal["strength", "weakness", "neutral"]
    explanation_quality: Optional[Literal[
        "correct_unprompted", "correct_prompted",
        "partially_correct", "incorrect", "vague_no_depth"
    ]] = None
    context: Optional[str] = None


class OwnershipSignal(BaseModel):
    type: Literal["ownership", "exposure"]
    quote: str
    context: str


class Evidence(BaseModel):
    technical_statements: List[EvidenceItem]
    ownership_signals: List[OwnershipSignal]
    problem_solving_approach: Optional[dict] = None
    communication_quality: Optional[dict] = None
    behavioral_signals: List[dict] = []
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
    recommendation: Literal["Strong Hire", "Hire", "Conditional", "Reject"]
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
