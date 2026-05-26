from src.models import DimensionScore, RoleMetadata, Decision, RoleLevel
from typing import List


class DecisionEngine:
    """
    Rule-based decision logic - NO AI.
    Deterministic, explainable, consistent.
    """

    THRESHOLDS = {
        RoleLevel.JUNIOR: 3.0,
        RoleLevel.INTERMEDIATE: 3.5,
        RoleLevel.SENIOR: 4.0,
        RoleLevel.LEAD: 4.2,
    }

    WEIGHTS = {
        RoleLevel.JUNIOR: {
            "Fundamentals": 0.35,
            "Coding": 0.30,
            "Communication": 0.15,
            "Learning": 0.15,
            "Practical": 0.05,
        },
        RoleLevel.INTERMEDIATE: {
            "Fundamentals": 0.25,
            "Coding": 0.25,
            "Architecture": 0.15,
            "Practical": 0.20,
            "Ownership": 0.10,
            "Communication": 0.05,
        },
        RoleLevel.SENIOR: {
            "Architecture": 0.25,
            "Fundamentals": 0.20,
            "Coding": 0.20,
            "Ownership": 0.15,
            "Practical": 0.15,
            "Communication": 0.05,
        },
        RoleLevel.LEAD: {
            "Architecture": 0.30,
            "Ownership": 0.25,
            "Fundamentals": 0.15,
            "Practical": 0.15,
            "Communication": 0.10,
            "Coding": 0.05,
        },
    }

    def make_decision(self, scores: List[DimensionScore], role: RoleMetadata) -> Decision:
        """
        Calculate weighted score and make hiring recommendation.

        Args:
            scores: List of dimension scores
            role: Role metadata

        Returns:
            Decision object
        """
        threshold = self.THRESHOLDS[role.role_level]
        weights = self.WEIGHTS[role.role_level]

        weighted_score = 0.0
        score_dict = {s.name: s.score for s in scores}

        for dimension, weight in weights.items():
            if dimension in score_dict:
                weighted_score += score_dict[dimension] * weight

        red_flag, red_flag_reason = self._check_red_flags(scores, role)
        if red_flag:
            return Decision(
                recommendation="Reject",
                reasoning=f"Critical red flag: {red_flag_reason}",
                weighted_score=round(weighted_score, 2),
                threshold=threshold,
            )

        if weighted_score >= threshold + 0.5:
            recommendation = "Strong Hire"
            reasoning = f"Score {weighted_score:.2f} significantly exceeds threshold {threshold}"
        elif weighted_score >= threshold:
            recommendation = "Hire"
            reasoning = f"Score {weighted_score:.2f} meets threshold {threshold}"
        elif weighted_score >= threshold - 2.0:
            recommendation = "Conditional"
            reasoning = (
                f"Score {weighted_score:.2f} below threshold {threshold} but above the "
                "conditional band — candidate shows enough substance to warrant a follow-up round or conditional offer"
            )
        else:
            recommendation = "Reject"
            reasoning = f"Score {weighted_score:.2f} below threshold {threshold}"

        return Decision(
            recommendation=recommendation,
            reasoning=reasoning,
            weighted_score=round(weighted_score, 2),
            threshold=threshold,
        )

    def _check_red_flags(self, scores: List[DimensionScore], role: RoleMetadata) -> tuple[bool, str]:
        """Check for critical red flags that result in automatic rejection."""
        score_dict = {s.name: s.score for s in scores}

        if role.role_level in [RoleLevel.INTERMEDIATE, RoleLevel.SENIOR, RoleLevel.LEAD]:
            if score_dict.get("Fundamentals", 0) < 2.0:
                return True, "Fundamentals score below minimum for experience level"

        # Coding red flag: only triggers when coding was attempted and critically failed (< 1.0).
        # A score of 1.0 means coding was not assessed in this interview — that is an assessment
        # gap, not a failure. Do not auto-reject candidates who simply weren't given coding tasks.
        if score_dict.get("Coding", 0) < 1.0:
            return True, "Coding attempted but critically failed — no meaningful output produced"

        if role.role_level in [RoleLevel.SENIOR, RoleLevel.LEAD]:
            if score_dict.get("Ownership", 0) < 2.0:
                return True, "Ownership score below minimum for senior role"

        return False, ""

    def _has_genuine_strengths(self, scores: List[DimensionScore]) -> bool:
        """Check if candidate demonstrates genuine strengths (≥ 3.0) in at least 2 dimensions."""
        strong_dimensions = [s for s in scores if s.score >= 3.0]
        return len(strong_dimensions) >= 2
