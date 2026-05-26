#!/usr/bin/env python3
"""
Measure evaluation accuracy against human ground truth.

Usage:
    # Single result vs its ground truth (auto-resolved):
    python scripts/measure_accuracy.py --result tests/results/v1.2/evaluation_abc.json

    # Explicit ground truth path:
    python scripts/measure_accuracy.py \
        --result tests/results/v1.2/evaluation_abc.json \
        --groundtruth tests/transcripts/my_transcript_groundtruth.json

    # Score all results in a version directory:
    python scripts/measure_accuracy.py --results-dir tests/results/v1.2 --transcripts-dir tests/transcripts
"""

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

ACCURACY_WEIGHTS = {
    "decision":        0.40,  # Most important: did we get hire/reject right?
    "issues_caught":   0.35,  # Did we catch every critical problem?
    "score_ranges":    0.15,  # Are dimension scores calibrated correctly?
    "strengths_found": 0.10,  # Did we recognize genuine strengths?
}


# ── Loading ───────────────────────────────────────────────────────────────────

def load_result(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def find_groundtruth(result: dict, transcripts_dir: Path) -> Path | None:
    transcript_file = result.get("transcript_file", "")
    if not transcript_file:
        return None
    transcript_path = Path(transcript_file)
    gt_path = transcript_path.parent / f"{transcript_path.stem}_groundtruth.json"
    if gt_path.exists():
        return gt_path
    # Try relative to transcripts_dir
    if transcripts_dir:
        alt = transcripts_dir / f"{transcript_path.stem}_groundtruth.json"
        if alt.exists():
            return alt
    return None


def load_groundtruth(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Accuracy checks ───────────────────────────────────────────────────────────

def check_decision(result: dict, gt: dict) -> dict:
    actual = result["decision"]["recommendation"]
    expected = gt["expected_decision"]
    match = actual == expected
    return {
        "pass": match,
        "actual": actual,
        "expected": expected,
        "score": 1.0 if match else 0.0,
        "detail": "✓ Correct decision" if match else f"✗ Expected {expected}, got {actual}",
    }


def check_issues_caught(result: dict, gt: dict) -> dict:
    critical_issues = [i for i in gt.get("critical_issues", []) if i.get("must_catch")]
    if not critical_issues:
        return {"pass": True, "score": 1.0, "caught": [], "missed": [], "detail": "No must-catch issues defined"}

    evidence = result.get("evidence", {})
    technical = evidence.get("technical_statements", [])
    ownership = evidence.get("ownership_signals", [])
    problem_solving = evidence.get("problem_solving_approach", {}) or {}
    comm_quality = evidence.get("communication_quality") or {}
    buzzwords = evidence.get("buzzwords_flagged", [])

    caught, missed = [], []
    for issue in critical_issues:
        if _issue_found_in_evidence(issue, technical, ownership, problem_solving, comm_quality, buzzwords):
            caught.append(issue["id"])
        else:
            missed.append(issue["id"])

    score = len(caught) / len(critical_issues)
    return {
        "pass": len(missed) == 0,
        "score": score,
        "caught": caught,
        "missed": missed,
        "total": len(critical_issues),
        "detail": f"Caught {len(caught)}/{len(critical_issues)} critical issues",
    }


def _matches_patterns(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def _fuzzy_match(text: str, pattern: str, threshold: float = 0.75) -> bool:
    """Check if pattern appears in text, tolerating 1-2 char STT substitutions.

    Slides a fixed-width window across text and checks SequenceMatcher ratio.
    When the text is shorter than the pattern (missing char case), compares
    the full text against the pattern directly. Skips patterns < 6 chars.
    """
    if pattern in text:
        return True
    if len(pattern) < 6:
        return False
    plen = len(pattern)
    tlen = len(text)
    if tlen < plen:
        # e.g. "singleton patern" (16) vs pattern "singleton pattern" (17)
        return tlen >= plen - 3 and SequenceMatcher(None, pattern, text).ratio() >= threshold
    for i in range(tlen - plen + 1):
        if SequenceMatcher(None, pattern, text[i : i + plen]).ratio() >= threshold:
            return True
    return False


def _matches_patterns_fuzzy(text: str, patterns: list[str]) -> bool:
    """Like _matches_patterns but fuzzy — use for raw transcript quote fields."""
    return any(_fuzzy_match(text, p) for p in patterns)


# ── Hallucination detection ───────────────────────────────────────────────────

def _quote_verifiable(quote: str, transcript_text: str, min_ngram: int = 4) -> bool:
    """Return True if at least one 4-word n-gram from the quote appears in the transcript.

    The model should only quote verbatim from the transcript, so any quoted phrase
    must have its core words traceable back to the source text. Short quotes (< 4 words)
    are skipped — too short to verify reliably.
    """
    words = quote.lower().split()
    if len(words) < min_ngram:
        return True  # too short to verify meaningfully
    transcript_lower = transcript_text.lower()
    for i in range(len(words) - min_ngram + 1):
        ngram = " ".join(words[i : i + min_ngram])
        if ngram in transcript_lower:
            return True
    return False


def load_transcript_for_result(result: dict) -> str | None:
    """Load the original transcript text referenced in an evaluation result."""
    path = result.get("transcript_file", "")
    if not path:
        return None
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else None


def check_hallucinations(result: dict, transcript_text: str) -> dict:
    """Verify that every extracted quote can be traced back to the transcript.

    Checks technical_statements and ownership_signals quotes. Returns a gate-style
    dict (no numeric score — either all quotes verify or they don't).
    """
    evidence = result.get("evidence", {})
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
        return {"pass": True, "flagged": [], "verified": 0, "total": 0,
                "detail": "No quotes to verify"}

    verified, flagged = [], []
    for item in to_check:
        if _quote_verifiable(item["quote"], transcript_text):
            verified.append(item)
        else:
            flagged.append(item)

    return {
        "pass": len(flagged) == 0,
        "flagged": flagged,
        "verified": len(verified),
        "total": len(to_check),
        "detail": f"Verified {len(verified)}/{len(to_check)} quotes against transcript",
    }


def _check_problem_solving_flag(issue: dict, problem_solving: dict) -> bool:
    flag = issue.get("check_flag")
    if not flag or flag not in problem_solving:
        return False
    expected = issue.get("expected_flag_value")
    return expected is None or problem_solving[flag] == expected


def _check_communication_quality(issue: dict, comm_quality: dict) -> bool:
    if not comm_quality:
        return False
    topic_patterns = [t.lower() for t in issue.get("topic_patterns", [])]
    if not any(t in ("communication",) for t in topic_patterns):
        return False
    return comm_quality.get("required_heavy_prompting", False) or \
        comm_quality.get("overall", "") in ("fragmented", "confused")


def _check_technical_statements(issue: dict, technical: list,
                                 quote_patterns: list[str], topic_patterns: list[str]) -> bool:
    expected_type = issue.get("expected_evidence_type")
    for stmt in technical:
        q = stmt.get("quote", "").lower()
        t = stmt.get("topic", "").lower()
        # Quotes are raw transcript text — use fuzzy to tolerate STT errors.
        # Topics are AI-interpreted — keep exact so we don't over-match.
        hit = (quote_patterns and _matches_patterns_fuzzy(q, quote_patterns)) or \
              (topic_patterns and _matches_patterns(t, topic_patterns))
        if hit and (not expected_type or stmt.get("evidence_type") == expected_type):
            return True
    return False


def _check_ownership_signals(issue: dict, ownership: list, quote_patterns: list[str]) -> bool:
    expected_type = issue.get("expected_ownership_type")
    if not expected_type:
        return False
    for sig in ownership:
        q = sig.get("quote", "").lower()
        quote_ok = _matches_patterns_fuzzy(q, quote_patterns) if quote_patterns else True
        if quote_ok and sig.get("type") == expected_type:
            return True
    return False


def _check_buzzwords(topic_patterns: list[str], buzzwords: list[str]) -> bool:
    buzzwords_lower = [b.lower() for b in buzzwords]
    return any(_matches_patterns(b, topic_patterns) for b in buzzwords_lower)


def _issue_found_in_evidence(issue: dict, technical: list, ownership: list,
                              problem_solving: dict, comm_quality: dict | None = None,
                              buzzwords: list | None = None) -> bool:
    quote_patterns = [p.lower() for p in issue.get("quote_patterns", [])]
    topic_patterns = [p.lower() for p in issue.get("topic_patterns", [])]

    return (
        _check_problem_solving_flag(issue, problem_solving)
        or _check_communication_quality(issue, comm_quality or {})
        or _check_technical_statements(issue, technical, quote_patterns, topic_patterns)
        or _check_ownership_signals(issue, ownership, quote_patterns)
        or (topic_patterns and _check_buzzwords(topic_patterns, buzzwords or []))
    )


def check_score_ranges(result: dict, gt: dict) -> dict:
    ranges = gt.get("expected_dimension_ranges", {})
    if not ranges:
        return {"pass": True, "score": 1.0, "in_range": [], "out_of_range": [], "detail": "No ranges defined"}

    score_map = {s["name"]: s["score"] for s in result.get("scores", [])}
    in_range = []
    out_of_range = []

    for dim, bounds in ranges.items():
        actual = score_map.get(dim)
        if actual is None:
            out_of_range.append(f"{dim}: not scored")
            continue
        lo, hi = bounds["min"], bounds["max"]
        if lo <= actual <= hi:
            in_range.append(f"{dim}: {actual} ✓")
        else:
            direction = "too high" if actual > hi else "too low"
            out_of_range.append(f"{dim}: {actual} (expected {lo}–{hi}, {direction})")

    score = len(in_range) / len(ranges) if ranges else 1.0
    return {
        "pass": len(out_of_range) == 0,
        "score": score,
        "in_range": in_range,
        "out_of_range": out_of_range,
        "detail": f"{len(in_range)}/{len(ranges)} scores within expected range",
    }


def _strength_found_in_statements(strength: dict, technical: list) -> bool:
    quote_patterns = [p.lower() for p in strength.get("quote_patterns", [])]
    topic_patterns = [p.lower() for p in strength.get("topic_patterns", [])]
    for stmt in technical:
        q = stmt.get("quote", "").lower()
        t = stmt.get("topic", "").lower()
        q_hit = any(p in q for p in quote_patterns)
        t_hit = any(p in t for p in topic_patterns)
        if (q_hit or t_hit) and stmt.get("evidence_type") == "strength":
            return True
    return False


def check_strengths_found(result: dict, gt: dict) -> dict:
    strengths = gt.get("known_strengths", [])
    if not strengths:
        return {"pass": True, "score": 1.0, "found": [], "missed": [], "detail": "No strengths defined"}

    technical = result.get("evidence", {}).get("technical_statements", [])
    found = [s["id"] for s in strengths if _strength_found_in_statements(s, technical)]
    missed = [s["id"] for s in strengths if s["id"] not in found]
    score = len(found) / len(strengths)
    return {
        "pass": len(missed) == 0,
        "score": score,
        "found": found,
        "missed": missed,
        "detail": f"Recognized {len(found)}/{len(strengths)} known strengths",
    }


# ── Overall scoring ───────────────────────────────────────────────────────────

def compute_accuracy(checks: dict) -> float:
    total = sum(
        ACCURACY_WEIGHTS[k] * checks[k]["score"]
        for k in ACCURACY_WEIGHTS
        if k in checks
    )
    return round(total * 100, 1)


# ── Display ───────────────────────────────────────────────────────────────────

_PASS = "[green]✓[/green]"
_FAIL = "[red]✗[/red]"


def _accuracy_color(accuracy: float) -> str:
    if accuracy >= 90:
        return "green"
    if accuracy >= 70:
        return "yellow"
    return "red"


def _print_check_table(checks: dict):
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Check", width=22)
    table.add_column("Weight", justify="right", width=8)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Weighted", justify="right", width=10)
    table.add_column("Detail", width=35)

    check_labels = {
        "decision":        "Decision",
        "issues_caught":   "Critical Issues",
        "score_ranges":    "Score Calibration",
        "strengths_found": "Strengths Found",
    }
    for key, label in check_labels.items():
        if key not in checks:
            continue
        c = checks[key]
        w = ACCURACY_WEIGHTS[key]
        s = c["score"]
        weighted = w * s * 100
        status = _PASS if c.get("pass") else _FAIL
        table.add_row(
            f"{status} {label}",
            f"{w * 100:.0f}%",
            f"{s * 100:.0f}%",
            f"{weighted:.1f}%",
            c.get("detail", ""),
        )
    console.print(table)


def _print_detail_sections(checks: dict, gt: dict):
    ic = checks.get("issues_caught", {})
    if ic.get("missed"):
        console.print("[bold red]Missed critical issues:[/bold red]")
        issue_map = {i["id"]: i for i in gt.get("critical_issues", [])}
        for iid in ic["missed"]:
            issue = issue_map.get(iid, {})
            console.print(f"  ✗ [{iid}] {issue.get('description', '')}")
        console.print()

    sr = checks.get("score_ranges", {})
    if sr.get("out_of_range"):
        console.print("[bold yellow]Scores outside expected range:[/bold yellow]")
        for item in sr["out_of_range"]:
            console.print(f"  ✗ {item}")
        console.print()

    sf = checks.get("strengths_found", {})
    if sf.get("missed"):
        console.print("[bold yellow]Strengths not recognized:[/bold yellow]")
        for iid in sf["missed"]:
            console.print(f"  ✗ {iid}")
        console.print()


def display_report(result: dict, gt: dict, checks: dict, accuracy: float,
                   result_path: Path, hallucination_check: dict | None = None):
    version = result.get("prompt_versions", {}).get("evidence_extraction", "unknown")
    ev_id = result.get("evaluation_id", "?")
    candidate = gt.get("candidate_name", "Unknown")
    sep = "=" * 65

    console.print(f"\n{sep}")
    console.print("[bold]ACCURACY REPORT[/bold]")
    console.print(sep)
    console.print(f"  Evaluation:  {ev_id}  (prompts {version})")
    console.print(f"  Candidate:   {candidate}")
    console.print(f"  Result file: {result_path.name}")
    console.print()

    color = _accuracy_color(accuracy)
    console.print(f"  [bold]Overall Accuracy: [{color}]{accuracy}%[/{color}][/bold]  (target: 90%)")
    console.print()

    _print_check_table(checks)
    _print_detail_sections(checks, gt)

    if hallucination_check:
        _print_hallucination_section(hallucination_check)

    console.print()
    accuracy_ok = accuracy >= 90
    hallucination_ok = hallucination_check is None or hallucination_check.get("pass", True)
    if accuracy_ok and hallucination_ok:
        console.print("[bold green]✓ PASS — meets 90% accuracy threshold[/bold green]")
    elif not accuracy_ok:
        gap = 90 - accuracy
        console.print(f"[bold red]✗ FAIL — {gap:.1f}% below threshold (need 90%, got {accuracy}%)[/bold red]")
    else:
        console.print("[bold green]✓ PASS (accuracy)[/bold green]  [bold red]✗ FAIL (hallucination gate)[/bold red]")

    console.print(f"{sep}\n")


def _print_hallucination_section(hc: dict):
    status = _PASS if hc.get("pass") else _FAIL
    console.print(f"  {status} Hallucination Gate  {hc['detail']}")
    if hc.get("flagged"):
        console.print("[bold red]  Unverifiable quotes (possible hallucinations):[/bold red]")
        for item in hc["flagged"]:
            console.print(f"    ✗ [{item['source']} / {item['label']}]  {item['quote'][:100]}")
        console.print()


# ── Batch mode ────────────────────────────────────────────────────────────────

def run_batch(results_dir: Path, transcripts_dir: Path) -> list[dict]:
    rows = []
    for result_path in sorted(results_dir.glob("evaluation_*.json")):
        result = load_result(result_path)
        gt_path = find_groundtruth(result, transcripts_dir)
        if not gt_path:
            console.print(f"[dim]Skipping {result_path.name} — no ground truth found[/dim]")
            continue
        gt = load_groundtruth(gt_path)
        checks = {
            "decision":        check_decision(result, gt),
            "issues_caught":   check_issues_caught(result, gt),
            "score_ranges":    check_score_ranges(result, gt),
            "strengths_found": check_strengths_found(result, gt),
        }
        accuracy = compute_accuracy(checks)
        transcript_text = load_transcript_for_result(result)
        hc = check_hallucinations(result, transcript_text) if transcript_text else None
        rows.append({
            "file": result_path.name,
            "version": result.get("prompt_versions", {}).get("evidence_extraction", "?"),
            "candidate": gt.get("candidate_name", "Unknown"),
            "decision_match": checks["decision"]["pass"],
            "issues_score": checks["issues_caught"]["score"],
            "accuracy": accuracy,
            "hallucination_pass": hc["pass"] if hc else None,
            "quotes_verified": f"{hc['verified']}/{hc['total']}" if hc else "n/a",
        })
    return rows


def display_batch(rows: list[dict], results_dir: Path):
    if not rows:
        console.print("[yellow]No evaluations with ground truth found.[/yellow]")
        return

    console.print(f"\n[bold]BATCH ACCURACY — {results_dir}[/bold]\n")
    table = Table(box=box.SIMPLE, header_style="bold dim")
    table.add_column("File", width=20)
    table.add_column("Ver", width=5)
    table.add_column("Candidate", width=18)
    table.add_column("Dec", justify="center", width=5)
    table.add_column("Issues", justify="right", width=7)
    table.add_column("Quotes", justify="right", width=9)
    table.add_column("Accuracy", justify="right", width=9)

    for r in rows:
        d_status = _PASS if r["decision_match"] else _FAIL
        i_str = f"{r['issues_score'] * 100:.0f}%"
        hp = r.get("hallucination_pass")
        if hp is None:
            q_str = "[dim]n/a[/dim]"
        else:
            q_str = f"{_PASS if hp else _FAIL} {r['quotes_verified']}"
        a_color = _accuracy_color(r["accuracy"])
        a_str = f"[{a_color}]{r['accuracy']}%[/{a_color}]"
        table.add_row(r["file"][:24], r["version"], r["candidate"][:22], d_status, i_str, q_str, a_str)

    console.print(table)
    avg = sum(r["accuracy"] for r in rows) / len(rows)
    color = _accuracy_color(avg)
    console.print(f"\n  Average accuracy: [{color}]{avg:.1f}%[/{color}]  ({len(rows)} evaluations scored)")
    if avg >= 90:
        console.print("  [bold green]✓ Phase 0 accuracy target MET[/bold green]")
    else:
        console.print(f"  [bold red]✗ Phase 0 target not met ({90 - avg:.1f}% gap)[/bold red]")
    console.print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Measure evaluation accuracy against ground truth")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result", type=Path, help="Single evaluation JSON to score")
    group.add_argument("--results-dir", type=Path, help="Score all evaluations in this directory")
    parser.add_argument("--groundtruth", type=Path, default=None, help="Ground truth JSON (auto-resolved if omitted)")
    parser.add_argument("--transcripts-dir", type=Path, default=Path("tests/transcripts"),
                        help="Directory to search for ground truth files (default: tests/transcripts)")
    args = parser.parse_args()

    if args.results_dir:
        rows = run_batch(args.results_dir, args.transcripts_dir)
        display_batch(rows, args.results_dir)
        return 0

    # Single result mode
    result_path = args.result
    if not result_path.exists():
        console.print(f"[red]Error: Result file not found: {result_path}[/red]")
        return 1

    result = load_result(result_path)

    gt_path = args.groundtruth or find_groundtruth(result, args.transcripts_dir)
    if not gt_path:
        console.print("[red]Error: No ground truth file found.[/red]")
        console.print("[dim]Create a _groundtruth.json file alongside the transcript,[/dim]")
        console.print("[dim]or pass --groundtruth <path>.[/dim]")
        return 1

    gt = load_groundtruth(gt_path)

    checks = {
        "decision":        check_decision(result, gt),
        "issues_caught":   check_issues_caught(result, gt),
        "score_ranges":    check_score_ranges(result, gt),
        "strengths_found": check_strengths_found(result, gt),
    }
    accuracy = compute_accuracy(checks)

    transcript_text = load_transcript_for_result(result)
    hc = check_hallucinations(result, transcript_text) if transcript_text else None
    if transcript_text is None:
        console.print("[dim]Note: transcript file not found — hallucination check skipped[/dim]")

    display_report(result, gt, checks, accuracy, result_path, hallucination_check=hc)
    hallucination_ok = hc is None or hc.get("pass", True)
    return 0 if (accuracy >= 90 and hallucination_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
