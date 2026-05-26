#!/usr/bin/env python3
"""
Compare evaluation results across prompt versions for the same transcript.

Usage:
    # Run fresh evaluations and compare:
    python scripts/compare_versions.py \
        --transcript tests/transcripts/my_transcript.txt \
        --versions v1.1 v1.2

    # Compare existing result JSONs (no API calls):
    python scripts/compare_versions.py \
        --results tests/results/v1.1/evaluation_abc.json tests/results/v1.2/evaluation_def.json

    # Run fresh, saving results alongside existing ones:
    python scripts/compare_versions.py \
        --transcript tests/transcripts/my_transcript.txt \
        --versions v1.1 v1.2 \
        --save
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine import InterviewEvaluationEngine
from src.ai_client import AIClient, RateLimitExhausted
from src.prompts import PromptManager
from src.decision_logic import DecisionEngine
from src.models import RoleMetadata, InterviewMetadata

load_dotenv()

console = Console()


# ── Loading helpers ───────────────────────────────────────────────────────────

def load_transcript(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_metadata(metadata_path: Path) -> tuple[RoleMetadata, InterviewMetadata]:
    with open(metadata_path) as f:
        data = json.load(f)
    return RoleMetadata(**data["role"]), InterviewMetadata(**data["interview"])


def load_groundtruth(transcript_path: Path) -> Optional[dict]:
    gt_path = transcript_path.parent / f"{transcript_path.stem}_groundtruth.json"
    if gt_path.exists():
        with open(gt_path) as f:
            return json.load(f)
    return None


def load_result_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def find_latest_result(version: str, transcript_file: str) -> Optional[Path]:
    """Find the most recently saved evaluation JSON for a given version and transcript."""
    results_dir = Path("tests/results") / version
    if not results_dir.exists():
        return None
    candidates = sorted(results_dir.glob("evaluation_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        with open(path) as f:
            data = json.load(f)
        if data.get("transcript_file") == transcript_file:
            return path
    return None


# ── Running evaluations ───────────────────────────────────────────────────────

def run_evaluation(transcript: str, role, interview, version: str, transcript_file: str) -> dict:
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    ai_client = AIClient(model=model)
    prompt_manager = PromptManager(version=version)
    decision_engine = DecisionEngine()
    engine = InterviewEvaluationEngine(ai_client, prompt_manager, decision_engine)
    result = engine.evaluate(
        transcript=transcript,
        role=role,
        interview=interview,
        transcript_file=transcript_file,
        include_interviewer_analysis=False,
    )
    return result.model_dump()


def save_result(result: dict, version: str) -> Path:
    output_dir = Path("tests/results") / version
    output_dir.mkdir(parents=True, exist_ok=True)
    ev_id = result["evaluation_id"]
    path = output_dir / f"evaluation_{ev_id}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    md_path = path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write(result["feedback_naresh"])
    return path


# ── Comparison display ────────────────────────────────────────────────────────

def _delta_str(base: float, new: float) -> str:
    delta = new - base
    if abs(delta) < 0.05:
        return "  —  "
    sign = "+" if delta > 0 else ""
    color = "green" if delta > 0 else "red"
    return f"[{color}]{sign}{delta:.1f}[/{color}]"


def display_comparison(results: dict[str, dict], ground_truth: Optional[dict]):
    versions = list(results.keys())
    base_version = versions[0]

    console.print("\n" + "=" * 70)
    console.print("[bold]PROMPT VERSION COMPARISON[/bold]")
    console.print("=" * 70 + "\n")

    # ── Dimension scores table ────────────────────────────────────────────────
    table = Table(box=box.SIMPLE, header_style="bold")
    table.add_column("Dimension", width=26)
    for v in versions:
        table.add_column(v, justify="right", width=8)
    if len(versions) > 1:
        table.add_column(f"Δ ({versions[-1]} vs {versions[0]})", justify="center", width=14)
    if ground_truth:
        table.add_column("Expected range", justify="center", width=16)

    all_dims = []
    for r in results.values():
        for s in r["scores"]:
            if s["name"] not in all_dims:
                all_dims.append(s["name"])

    expected_ranges = ground_truth.get("expected_dimension_ranges", {}) if ground_truth else {}

    for dim in all_dims:
        row = [dim]
        scores_for_dim = []
        for v in versions:
            score_map = {s["name"]: s["score"] for s in results[v]["scores"]}
            val = score_map.get(dim)
            if val is not None:
                scores_for_dim.append(val)
                # Colour against expected range
                if dim in expected_ranges:
                    lo = expected_ranges[dim]["min"]
                    hi = expected_ranges[dim]["max"]
                    if lo <= val <= hi:
                        row.append(f"[green]{val:.1f}[/green]")
                    else:
                        row.append(f"[red]{val:.1f}[/red]")
                else:
                    row.append(f"{val:.1f}")
            else:
                scores_for_dim.append(None)
                row.append("N/A")

        if len(versions) > 1:
            base_s = {s["name"]: s["score"] for s in results[base_version]["scores"]}.get(dim)
            last_s = {s["name"]: s["score"] for s in results[versions[-1]]["scores"]}.get(dim)
            if base_s is not None and last_s is not None:
                row.append(_delta_str(base_s, last_s))
            else:
                row.append("  —  ")

        if ground_truth:
            if dim in expected_ranges:
                lo = expected_ranges[dim]["min"]
                hi = expected_ranges[dim]["max"]
                row.append(f"{lo}–{hi}")
            else:
                row.append("—")

        table.add_row(*row)

    # Weighted score row
    row = ["[bold]Weighted Score[/bold]"]
    ws_values = []
    for v in versions:
        ws = results[v]["decision"]["weighted_score"]
        ws_values.append(ws)
        row.append(f"[bold]{ws:.2f}[/bold]")
    if len(versions) > 1:
        row.append(_delta_str(ws_values[0], ws_values[-1]))
    if ground_truth:
        row.append("—")
    table.add_row(*row)

    console.print(table)

    # ── Decision row ──────────────────────────────────────────────────────────
    console.print("[bold]Decision:[/bold]")
    gt_decision = ground_truth.get("expected_decision") if ground_truth else None
    for v, r in results.items():
        rec = r["decision"]["recommendation"]
        match_indicator = ""
        if gt_decision:
            match_indicator = " [green]✓[/green]" if rec == gt_decision else " [red]✗[/red]"
        console.print(f"  {v}: [bold]{rec}[/bold]{match_indicator}  ({r['decision']['weighted_score']:.2f})")
    if gt_decision:
        console.print(f"  Ground truth: [bold]{gt_decision}[/bold]")

    # ── Processing times ──────────────────────────────────────────────────────
    console.print("\n[bold]Processing time:[/bold]")
    for v, r in results.items():
        console.print(f"  {v}: {r.get('processing_time_seconds', '?')}s")

    # ── Key evidence differences ──────────────────────────────────────────────
    if len(versions) == 2:
        v1, v2 = versions
        console.print(f"\n[bold]Score movements ({v1} → {v2}):[/bold]")
        s1 = {s["name"]: s["score"] for s in results[v1]["scores"]}
        s2 = {s["name"]: s["score"] for s in results[v2]["scores"]}
        moved = [(d, s1.get(d, 0), s2.get(d, 0)) for d in all_dims if abs(s2.get(d, 0) - s1.get(d, 0)) >= 0.3]
        if moved:
            for dim, old, new in sorted(moved, key=lambda x: abs(x[2] - x[1]), reverse=True):
                delta = new - old
                sign = "+" if delta > 0 else ""
                color = "green" if delta > 0 else "red"
                console.print(f"  {dim}: {old:.1f} → {new:.1f}  [{color}]({sign}{delta:.1f})[/{color}]")
        else:
            console.print("  No dimension moved by ≥ 0.3")

    # ── Ground truth critical issues ──────────────────────────────────────────
    if ground_truth and ground_truth.get("critical_issues"):
        console.print("\n[bold]Critical issue detection:[/bold]")
        # Import the checker from measure_accuracy
        from measure_accuracy import check_issues_caught
        for v, r in results.items():
            ic = check_issues_caught(r, ground_truth)
            caught_str = ", ".join(ic["caught"]) or "none"
            missed_str = ", ".join(ic["missed"]) or "none"
            console.print(f"  {v}: caught {len(ic['caught'])}/{ic['total']}  "
                          f"([green]{caught_str}[/green]" +
                          (f" | [red]missed: {missed_str}[/red]" if ic["missed"] else "") + ")")

    console.print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare evaluations across prompt versions")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--transcript", type=Path, help="Transcript to evaluate with each version")
    mode.add_argument("--results", type=Path, nargs="+", help="Existing evaluation JSONs to compare")

    parser.add_argument("--versions", nargs="+", help="Prompt versions (with --transcript)")
    parser.add_argument("--save", action="store_true", help="Save new evaluation results to disk")
    parser.add_argument("--reuse", action="store_true",
                        help="Load most recent saved result for each version instead of re-running")
    args = parser.parse_args()

    results: dict[str, dict] = {}
    ground_truth = None

    if args.results:
        # Load provided result files directly
        for path in args.results:
            if not path.exists():
                console.print(f"[red]Error: {path} not found[/red]")
                return 1
            data = load_result_json(path)
            label = data.get("prompt_versions", {}).get("evidence_extraction", path.stem)
            results[label] = data
        # Try to find ground truth from first result
        transcript_file = next(iter(results.values())).get("transcript_file", "")
        if transcript_file:
            gt_path = Path(transcript_file).parent / f"{Path(transcript_file).stem}_groundtruth.json"
            if gt_path.exists():
                with open(gt_path) as f:
                    ground_truth = json.load(f)

    else:
        # Transcript + versions mode
        if not args.versions:
            console.print("[red]Error: --versions required when using --transcript[/red]")
            return 1

        transcript_path = args.transcript
        if not transcript_path.exists():
            console.print(f"[red]Error: Transcript not found: {transcript_path}[/red]")
            return 1

        metadata_path = transcript_path.parent / f"{transcript_path.stem}_metadata.json"
        if not metadata_path.exists():
            console.print(f"[red]Error: Metadata not found: {metadata_path}[/red]")
            return 1

        role, interview = load_metadata(metadata_path)
        ground_truth = load_groundtruth(transcript_path)
        transcript = None  # Lazy load — only needed if running evaluations

        for version in args.versions:
            if args.reuse:
                saved = find_latest_result(version, str(transcript_path))
                if saved:
                    console.print(f"[dim]Reusing existing result for {version}: {saved.name}[/dim]")
                    results[version] = load_result_json(saved)
                    continue

            if transcript is None:
                transcript = load_transcript(transcript_path)

            console.print(f"\n[cyan]Running {version}...[/cyan]")
            try:
                data = run_evaluation(transcript, role, interview, version, str(transcript_path))
                results[version] = data
                if args.save:
                    saved = save_result(data, version)
                    console.print(f"  Saved to: {saved}")
            except RateLimitExhausted as e:
                console.print(f"[red]  Rate limit hit for {version}: {e}[/red]")
            except Exception as e:
                console.print(f"[red]  Failed for {version}: {e}[/red]")

    if len(results) < 2:
        console.print("[yellow]Need at least 2 results to compare.[/yellow]")
        if len(results) == 1:
            v, r = next(iter(results.items()))
            console.print(f"\nSingle result — {v}: {r['decision']['recommendation']} ({r['decision']['weighted_score']:.2f})")
        return 1

    display_comparison(results, ground_truth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
