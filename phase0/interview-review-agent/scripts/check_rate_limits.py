#!/usr/bin/env python3
"""
Check current OpenAI rate limits for the configured model.

Makes a 1-token API call and reads the rate limit headers from the response.

Usage:
    python scripts/check_rate_limits.py
    python scripts/check_rate_limits.py --model gpt-4o-mini
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_client import AIClient, AIClientError

load_dotenv()

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Check current OpenAI rate limits")
    parser.add_argument("--model", type=str, default=None, help="Model to check (default: gpt-4o)")
    args = parser.parse_args()

    import os
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o")

    console.print(f"\n[cyan]Checking rate limits for model:[/cyan] {model}")
    console.print("[dim]Making a 1-token probe call to read response headers...[/dim]\n")

    try:
        client = AIClient(model=model)
        rl = client.check_rate_limits()
    except AIClientError as e:
        console.print(f"[red]Failed to check rate limits: {e}[/red]")
        return 1

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim", width=30)
    table.add_column("Remaining", justify="right")
    table.add_column("Limit", justify="right")
    table.add_column("Used %", justify="right")
    table.add_column("Resets In", justify="right")

    def _pct(remaining, limit) -> str:
        if remaining is None or limit is None or limit == 0:
            return "N/A"
        used = round((1 - remaining / limit) * 100)
        color = "green" if used < 50 else "yellow" if used < 80 else "red"
        return f"[{color}]{used}%[/{color}]"

    table.add_row(
        "Requests per minute",
        str(rl.remaining_requests) if rl.remaining_requests is not None else "N/A",
        str(rl.limit_requests) if rl.limit_requests is not None else "N/A",
        _pct(rl.remaining_requests, rl.limit_requests),
        rl.reset_requests or "N/A",
    )
    table.add_row(
        "Tokens per minute",
        str(rl.remaining_tokens) if rl.remaining_tokens is not None else "N/A",
        str(rl.limit_tokens) if rl.limit_tokens is not None else "N/A",
        _pct(rl.remaining_tokens, rl.limit_tokens),
        rl.reset_tokens or "N/A",
    )

    console.print(table)

    # Advisory
    if rl.remaining_requests is not None and rl.remaining_requests < 5:
        console.print(f"\n[red]Warning: only {rl.remaining_requests} requests remaining. "
                      f"Resets in {rl.reset_requests}.[/red]")
    elif rl.remaining_tokens is not None and rl.limit_tokens and (rl.remaining_tokens / rl.limit_tokens) < 0.1:
        console.print(f"\n[yellow]Warning: token budget is low ({rl.remaining_tokens} remaining). "
                      f"Resets in {rl.reset_tokens}.[/yellow]")
    else:
        console.print("\n[green]Rate limits look healthy.[/green]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
