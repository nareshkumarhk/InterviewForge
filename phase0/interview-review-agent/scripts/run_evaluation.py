#!/usr/bin/env python3
"""
Main script to run interview evaluation.

Usage:
    python scripts/run_evaluation.py --transcript tests/transcripts/transcript_1.txt --version v1.0
"""

import argparse
import json
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import InterviewEvaluationEngine
from src.ai_client import AIClient, RateLimitExhausted
from src.prompts import PromptManager
from src.decision_logic import DecisionEngine
from src.models import RoleMetadata, InterviewMetadata

load_dotenv()

import os
console = Console()


def load_transcript(transcript_path: Path) -> str:
    with open(transcript_path, "r", encoding="utf-8") as f:
        return f.read()


def load_metadata(metadata_path: Path) -> tuple[RoleMetadata, InterviewMetadata]:
    with open(metadata_path, "r") as f:
        data = json.load(f)
    role = RoleMetadata(**data["role"])
    interview = InterviewMetadata(**data["interview"])
    return role, interview


def _make_output_stem(transcript_path: Path, role: RoleMetadata, evaluation_id: str) -> str:
    """Derive a human-readable output filename from the transcript and role.

    Example: L1_Interview_Nimil_E_Raveendran_Tech_Lead_Net_lead_2dd89391
    """
    name = transcript_path.name
    # Strip known recording extensions from the right (iteratively)
    for ext in ('.txt', '.transcript', '.ordered', '.mp4', '.docx', '.wav', '.vtt'):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
    # Remove recording noise that adds no candidate/role context
    name = re.sub(r'-\d{8}_\d{6}', '', name)                               # -20260518_135855
    name = re.sub(r'[\s_-]*Meeting\s+Recording', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[\s_-]*DBiz\.?ai', '', name, flags=re.IGNORECASE)
    # Sanitize: replace any non-alphanumeric run with a single underscore
    name = re.sub(r'[^\w]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    # Truncate at a word boundary (keep it filesystem-friendly but readable)
    if len(name) > 60:
        name = name[:60].rstrip('_')
    role_slug = role.role_level.value  # "lead" | "senior" | "intermediate" | "junior"
    return f"{name}_{role_slug}_{evaluation_id}"


def save_result(result, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(result.model_dump(), f, indent=2)

    md_path = output_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write(result.feedback_naresh)

    console.print(f"\n[green]Results saved:[/green]")
    console.print(f"  - JSON: {json_path}")
    console.print(f"  - Markdown: {md_path}")


def _warn_hallucinations(result, transcript: str, con: Console) -> None:
    """Check every extracted quote against the transcript and warn if unverifiable."""
    evidence = result.model_dump().get("evidence", {})
    to_check = []
    for stmt in evidence.get("technical_statements", []):
        q = stmt.get("quote", "").strip()
        if q:
            to_check.append({"quote": q, "label": stmt.get("topic", "?"), "source": "technical"})
    for sig in evidence.get("ownership_signals", []):
        q = sig.get("quote", "").strip()
        if q:
            to_check.append({"quote": q, "label": sig.get("type", "?"), "source": "ownership"})

    if not to_check:
        return

    transcript_lower = transcript.lower()
    flagged = []
    for item in to_check:
        words = item["quote"].lower().split()
        if len(words) < 4:
            continue
        verified = any(
            " ".join(words[i : i + 4]) in transcript_lower
            for i in range(len(words) - 3)
        )
        if not verified:
            flagged.append(item)

    if flagged:
        con.print("\n[bold red]⚠ Hallucination warning — quotes not traceable to transcript:[/bold red]")
        for item in flagged:
            con.print(f"  ✗ [{item['source']} / {item['label']}]  {item['quote'][:100]}")
        con.print("[dim]These quotes could not be verified against the source transcript.[/dim]\n")


def main():
    parser = argparse.ArgumentParser(description="Run interview evaluation")
    parser.add_argument("--transcript", type=Path, required=True, help="Path to transcript file")
    parser.add_argument("--version", type=str, default="v1.0", help="Prompt version to use (default: v1.0)")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: tests/results/{version}/evaluation_{id})")
    parser.add_argument("--interviewer-analysis", action="store_true", help="Include interviewer quality analysis")

    args = parser.parse_args()

    if not args.transcript.exists():
        console.print(f"[red]Error: Transcript file not found: {args.transcript}[/red]")
        return 1

    console.print(f"\n[cyan]Loading transcript:[/cyan] {args.transcript}")
    transcript = load_transcript(args.transcript)

    metadata_path = args.transcript.parent / f"{args.transcript.stem}_metadata.json"
    if not metadata_path.exists():
        console.print(f"[red]Error: Metadata file not found: {metadata_path}[/red]")
        console.print("[yellow]Expected format: {transcript_name}_metadata.json[/yellow]")
        return 1

    role, interview = load_metadata(metadata_path)

    console.print(f"\n[cyan]Initializing evaluation engine[/cyan] (prompts: {args.version})...")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    ai_client = AIClient(model=model)
    prompt_manager = PromptManager(version=args.version)
    decision_engine = DecisionEngine()
    engine = InterviewEvaluationEngine(ai_client, prompt_manager, decision_engine)

    console.print(f"\n[cyan]Processing evaluation...[/cyan]\n")
    try:
        result = engine.evaluate(
            transcript=transcript,
            role=role,
            interview=interview,
            transcript_file=str(args.transcript),
            include_interviewer_analysis=args.interviewer_analysis,
        )
    except RateLimitExhausted as e:
        console.print("\n[red]Rate limit exhausted — could not complete evaluation.[/red]")
        rl = e.rate_limit
        if rl:
            console.print("\n[bold]Current rate limit state:[/bold]")
            console.print(f"  Requests: {rl.remaining_requests}/{rl.limit_requests} remaining")
            console.print(f"  Tokens:   {rl.remaining_tokens}/{rl.limit_tokens} remaining")
            console.print("\n[bold]Retry in:[/bold]")
            console.print(f"  Requests reset in: [yellow]{rl.reset_requests}[/yellow]")
            console.print(f"  Tokens reset in:   [yellow]{rl.reset_tokens}[/yellow]")
        console.print("\n[dim]Check tests/results/partial/ for any partial results saved.[/dim]")
        console.print("\n[cyan]Re-run the same command once the limit resets.[/cyan]")
        return 1

    if args.output:
        output_path = args.output
    else:
        output_dir = Path("tests/results") / args.version
        stem = _make_output_stem(args.transcript, role, result.evaluation_id)
        output_path = output_dir / stem

    save_result(result, output_path)

    _warn_hallucinations(result, transcript, console)

    console.print("\n" + "=" * 70)
    console.print("[bold]EVALUATION SUMMARY[/bold]")
    console.print("=" * 70 + "\n")
    console.print(f"Evaluation ID:  {result.evaluation_id}")
    console.print(f"Role:           {role.role_title} ({role.role_level.value})")
    console.print(f"Decision:       [bold]{result.decision.recommendation}[/bold]")
    console.print(f"Weighted Score: {result.decision.weighted_score:.2f} (threshold: {result.decision.threshold})")
    console.print(f"Processing:     {result.processing_time_seconds}s")

    if ai_client.last_rate_limit:
        rl = ai_client.last_rate_limit
        console.print("\n[bold]Rate Limit (after last call):[/bold]")
        console.print(f"  Requests:  {rl.remaining_requests}/{rl.limit_requests} remaining  (resets {rl.reset_requests})")
        console.print(f"  Tokens:    {rl.remaining_tokens}/{rl.limit_tokens} remaining  (resets {rl.reset_tokens})")

    console.print("\n[bold]Dimension Scores:[/bold]")
    for score in result.scores:
        bar = "#" * int(score.score * 2)
        console.print(f"  {score.name:<25} {score.score:.1f}/5  {bar}")

    console.print("\n[bold]Quick Summary:[/bold]")
    console.print(Markdown(result.feedback_quick))

    console.print("\n[bold]Full Feedback (Naresh-style):[/bold]")
    console.print(Markdown(result.feedback_naresh))

    return 0


if __name__ == "__main__":
    sys.exit(main())
