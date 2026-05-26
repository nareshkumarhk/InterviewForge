"""
Transcript compression — two stages:
  1. Rule-based: splits Teams/Zoom auto-transcript concatenated-line format into
     attributed speaker turns, removes timestamps, filler words, and ack-only lines.
  2. AI-based: optional second pass to further condense verbose phrasing while
     keeping all technical content. Only triggered when stage 1 is not enough.
"""

import re
from typing import List, Tuple

# ── Speaker name pattern ───────────────────────────────────────────────────────
# Matches "Naresh Kumar", "Manas Ranjan Jena", etc. (2–4 capitalised words)
_NAME = r"[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+){1,3}"

# Teams has two new-turn formats depending on position:
#   Format A: "Name H:MM Name X minutes Y seconds"  (first turn / after system event)
#   Format B: "Name X minutes Y seconds H:MM Name X minutes Y seconds"  (subsequent turns)
_NEW_TURN_A = re.compile(
    rf"({_NAME})\s+\d{{1,2}}:\d{{2}}\s+\1\s+\d+ minutes? \d+ seconds?\s*"
)
_NEW_TURN_B = re.compile(
    rf"({_NAME})\s+\d+ minutes? \d+ seconds?\s+\d{{1,2}}:\d{{2}}\s+\1\s+\d+ minutes? \d+ seconds?\s*"
)

# Continuation header — single name + long timestamp only (no doubled name)
_CONTINUATION = re.compile(
    rf"({_NAME})\s+\d+ minutes? \d+ seconds?\s*"
)

_TURN_REPLACEMENT = r"\n\1: "

_FILLER = re.compile(
    r"\b(um+|uh+|hmm+|err?|like(?=\s)|you know|sort of|kind of|basically"
    r"|literally|right\s*\?|okay\s*so|so\s+so)\b",
    re.IGNORECASE,
)

_ACK_ONLY = re.compile(
    r"^\s*(yeah[.,!?]*|yes[.,!?]*|okay[.,!?]*|ok[.,!?]*|sure[.,!?]*"
    r"|right[.,!?]*|m+[.,!?]*|hmm+[.,!?]*|uh[- ]?huh[.,!?]*|no+[.,!?]*"
    r"|good[.,!?]*|great[.,!?]*)\s*$",
    re.IGNORECASE,
)

# Teams "started transcription" system event
_SYSTEM_EVENT = re.compile(r"started transcription", re.IGNORECASE)

# Unicode private-use area characters (Teams icons/emoji) — U+E000 to U+F8FF
_PRIVATE_UNICODE = re.compile(r"[-]")


# ── Public API ─────────────────────────────────────────────────────────────────

def compress_rules(transcript: str) -> Tuple[str, int, int]:
    """
    Rule-based compression of a Teams-style auto-transcript.

    Returns:
        (compressed_text, original_token_estimate, new_token_estimate)
    """
    original_tokens = _est(transcript)
    text = _strip_noise(transcript)
    text = _split_into_turns(text)
    text = _remove_filler_and_acks(text)
    text = _dedupe_lines(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, original_tokens, _est(text)


_AI_COMPRESS_SYSTEM = (
    "You are compressing an interview transcript to reduce token count.\n\n"
    "PRESERVE — do not remove or paraphrase:\n"
    "- Every technical claim or explanation the candidate made\n"
    "- Ownership/exposure language: 'I designed', 'we used', 'the team decided'\n"
    "- All technology names and claimed experience levels\n"
    "- Behavioral signals: disagreement handling, giving feedback, working under pressure\n"
    "- All interviewer questions (shorten phrasing but keep intent)\n"
    "- Any code discussed, traced, or written\n\n"
    "REMOVE:\n"
    "- Filler words: um, uh, basically, like, you know, sort of\n"
    "- Pure acknowledgment exchanges: 'Yeah', 'Okay', 'Sure', 'Right'\n"
    "- Repetition of the same point\n"
    "- Timestamps and speaker-name metadata\n\n"
    "Output ONLY the compressed transcript, no commentary."
)

# gpt-4o-mini context window is 128K tokens.
# Leave ~4K for system prompt + user prompt prefix + output overhead.
_AI_CHUNK_TOKENS = 20_000


def compress_ai(transcript: str, ai_client, target_tokens: int) -> Tuple[str, int]:
    """
    AI-based compression preserving all technical content and ownership signals.

    Splits into ~20K-token chunks when the transcript exceeds the model's safe
    single-call size, compresses each proportionally, then reassembles.

    Returns:
        (compressed_text, new_token_estimate)
    """
    total_tokens = _est(transcript)

    if total_tokens <= _AI_CHUNK_TOKENS:
        compressed = _compress_ai_single(transcript, ai_client, target_tokens)
        return compressed, _est(compressed)

    # Chunked path: split on speaker-turn boundaries, group into ~20K chunks
    chunks = _split_chunks(transcript, _AI_CHUNK_TOKENS)
    chunk_target = max(500, target_tokens // len(chunks))
    compressed_parts = [
        _compress_ai_single(chunk, ai_client, chunk_target)
        for chunk in chunks
    ]
    compressed = "\n".join(p for p in compressed_parts if p.strip())
    return compressed, _est(compressed)


def _compress_ai_single(transcript: str, ai_client, target_tokens: int) -> str:
    target_chars = target_tokens * 4
    user_prompt = (
        f"Compress to approximately {target_tokens:,} tokens ({target_chars:,} chars). "
        f"Preserve all technical content.\n\n{transcript}"
    )
    return ai_client.complete(_AI_COMPRESS_SYSTEM, user_prompt, temperature=0.1)


def _split_chunks(text: str, chunk_tokens: int) -> List[str]:
    """Split text on speaker-turn lines into chunks of at most chunk_tokens each."""
    lines = text.splitlines(keepends=True)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = _est(line)
        if current_tokens + line_tokens > chunk_tokens and current:
            chunks.append("".join(current).strip())
            current = []
            current_tokens = 0
        current.append(line)
        current_tokens += line_tokens

    if current:
        chunks.append("".join(current).strip())
    return chunks


# ── Internal helpers ──────────────────────────────────────────────────────────

def _strip_noise(text: str) -> str:
    """Remove Unicode private-use icons and 'Name started transcription' system events.
    Timestamps are left intact — they are needed by _split_into_turns."""
    text = _PRIVATE_UNICODE.sub("", text)
    # Remove the whole "Name started transcription" phrase (not just the verb)
    text = re.sub(rf"\b{_NAME}\s+started transcription\b", "", text, flags=re.IGNORECASE)
    # Remove bare bracket-only timestamp lines: "[X minutes Y seconds] |"
    text = re.sub(r"\[?\d+ minutes? \d+ seconds?\]?\s*\|?\s*\n", "\n", text, flags=re.IGNORECASE)
    return text


def _split_into_turns(text: str) -> str:
    """
    Replace inline speaker headers with newline-prefixed 'Speaker: ' markers,
    strip timestamps, then deduplicate consecutive turns by the same speaker.
    """
    # 1. Format B first (longer pattern — must precede A and continuation)
    text = _NEW_TURN_B.sub(_TURN_REPLACEMENT, text)
    # 2. Format A
    text = _NEW_TURN_A.sub(_TURN_REPLACEMENT, text)
    # 3. Remaining continuation headers (single name + long timestamp only)
    text = _CONTINUATION.sub(_TURN_REPLACEMENT, text)
    # 4. Strip any remaining long timestamps
    text = re.sub(r"\d+ minutes? \d+ seconds?", "", text, flags=re.IGNORECASE)
    # 5. Strip short HH:MM timestamps
    text = re.sub(r"\b\d{1,2}:\d{2}\b", "", text)
    # 6. Normalise whitespace within lines
    lines = [re.sub(r"  +", " ", ln).strip() for ln in text.splitlines()]
    return _dedupe_speakers(lines)


def _dedupe_speakers(lines: List[str]) -> str:
    """Merge consecutive turns from the same speaker into one."""
    out: List[str] = []
    last_speaker = ""

    for line in lines:
        if not line:
            continue
        m = re.match(rf"^({_NAME}):\s*(.*)", line)
        if not m:
            out.append(line)
            continue
        speaker, content = m.group(1).strip(), m.group(2).strip()
        if speaker == last_speaker:
            if content:
                out.append(content)
        else:
            last_speaker = speaker
            if content:
                out.append(f"{speaker}: {content}")

    return "\n".join(out)


def _remove_filler_and_acks(text: str) -> str:
    """Strip filler words and drop pure-acknowledgment lines."""
    text = _FILLER.sub("", text)
    lines = [ln for ln in text.splitlines() if not _ACK_ONLY.match(ln.strip())]
    return "\n".join(lines)


def _dedupe_lines(text: str) -> str:
    """
    Drop exact-duplicate lines while preserving order.

    The Teams .ordered.transcript format encodes each utterance in 5-6 different
    format views. After speaker-turn extraction they collapse to identical lines,
    but leftover tokens (initials, bare timestamps) can break consecutive-speaker
    deduplication. This pass cleans up whatever remains.
    """
    seen: set = set()
    out: List[str] = []
    for line in text.splitlines():
        if line not in seen:
            seen.add(line)
            out.append(line)
    return "\n".join(out)


def _est(text: str) -> int:
    return len(text) // 4
