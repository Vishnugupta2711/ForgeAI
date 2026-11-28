"""Deterministic verification checks for ForgeAI Judge."""

from __future__ import annotations

import difflib
import json
import math
import re
from datetime import datetime

from forge.agent.schema import AnswerRecord
from forge.evals.models import ExpectedBehavior
from forge.ingest.models import Chunk

_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(
    r"(?<![a-zA-Z0-9_])(?<![a-zA-Z]-)(\d+(?:,\d{3})*(?:\.\d+)?)(?![a-zA-Z0-9_])"
)
_CURRENCY_PREFIX_RE = re.compile(r"([$€£])\s*(\d+(?:,\d{3})*(?:\.\d+)?)")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MONTH_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)

_UNIT_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:metres|meters|meter|m)\b"), r"\1 m"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:kilometers|kilometres|km)\b"), r"\1 km"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:days|day)\b"), r"\1 day"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:minutes|minute|mins|min)\b"), r"\1 min"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b"), r"\1 hour"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:volts|volt|v)\b"), r"\1 v"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:percent|%)\b"), r"\1 percent"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*%"), r"\1 percent"),
]

_CURRENCY_MAP = {"$": "usd", "€": "eur", "£": "gbp"}
_MONTH_FORMATS = ("%B %d %Y", "%b %d %Y")
_SNAKE_UNDERSCORE_RE = re.compile(r"(?<=[a-zA-Z])_(?=[a-zA-Z])")
_ABBR_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bmoq\b"), "minimum order quantity"),
    (re.compile(r"(?<!\d )\bmin\b"), "minimum"),
    (re.compile(r"\bmax\b"), "maximum"),
    (re.compile(r"\bqty\b"), "quantity"),
    (re.compile(r"\bpct\b"), "percent"),
    (re.compile(r"\bavg\b"), "average"),
    (re.compile(r"\bstd\b"), "standard"),
    (re.compile(r"\best\b"), "estimated"),
    (re.compile(r"\bapprox\b"), "approximately"),
]


def _canon_number(raw: str) -> str:
    cleaned = raw.replace(",", "")
    try:
        val = float(cleaned)
        if val.is_integer():
            return str(int(val))
        return f"{val:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return cleaned


def _canon_dates(text: str) -> str:
    def _iso_repl(m: re.Match[str]) -> str:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    def _slash_repl(m: re.Match[str]) -> str:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= a <= 12 and 1 <= b <= 31:
            return f"{y:04d}-{a:02d}-{b:02d}"
        return m.group(0)

    def _month_repl(m: re.Match[str]) -> str:
        mon, day, yr = m.group(1), m.group(2), m.group(3)
        mon_norm = "Sep" if mon.lower() == "sept" else mon
        for fmt in _MONTH_FORMATS:
            try:
                dt = datetime.strptime(f"{mon_norm} {day} {yr}", fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return m.group(0)

    text = _ISO_DATE_RE.sub(_iso_repl, text)
    text = _SLASH_DATE_RE.sub(_slash_repl, text)
    text = _MONTH_DATE_RE.sub(_month_repl, text)
    return text


def _unwrap_pdf_hyphens(text: str) -> str:
    return re.sub(r"([A-Za-z0-9])-\s*\n\s*([A-Za-z0-9])", r"\1-\2", text)


def canonicalize_text(text: str) -> str:
    """Canonicalize numbers, currency, units, dates, and whitespace for matching."""
    out = _canon_dates(_unwrap_pdf_hyphens(text))
    out = _CURRENCY_PREFIX_RE.sub(
        lambda m: f"{_CURRENCY_MAP.get(m.group(1), m.group(1))} {_canon_number(m.group(2))}",
        out,
    )
    out = out.lower()
    out = _SNAKE_UNDERSCORE_RE.sub(" ", out)
    out = _NUMBER_RE.sub(lambda m: _canon_number(m.group(1)), out)
    for pattern, repl in _UNIT_REPLACEMENTS:
        out = pattern.sub(repl, out)
    for pattern, repl in _ABBR_REPLACEMENTS:
        out = pattern.sub(repl, out)
    out = _WHITESPACE_RE.sub(" ", out).strip()
    return out


def check_refusal(expected_behavior: ExpectedBehavior, refused: bool) -> tuple[bool, str]:
    """Deterministically verify refusal status against expected_behavior."""
    if expected_behavior == "refuse":
        if refused:
            return True, "Correctly refused out-of-scope question."
        return False, "Expected refusal for out-of-scope question, but agent answered."
    if refused:
        return False, f"Wrongful refusal: expected '{expected_behavior}'."
    return True, "Refusal check passed."


def verify_quote_in_evidence(
    quote: str,
    evidence_texts: list[str],
    *,
    min_chars: int = 4,
) -> bool:
    """Verify quote is a real exact or whitespace-normalized substring of evidence."""
    cleaned = _unwrap_pdf_hyphens(quote).strip()
    if len(cleaned) < min_chars:
        return False
    norm_quote = _WHITESPACE_RE.sub(" ", cleaned).lower()
    for text in evidence_texts:
        if cleaned in text or quote.strip() in text:
            return True
        norm_ev = _WHITESPACE_RE.sub(" ", _unwrap_pdf_hyphens(text)).lower()
        if norm_quote in norm_ev:
            return True
    return False


def check_citation_validity(
    record: AnswerRecord,
    chunk_map: dict[str, Chunk],
) -> tuple[bool, list[str]]:
    """Verify cited doc_id exists and each cited quote matches an actually retrieved chunk."""
    out = record.output
    if out.refused:
        return True, []

    doc_ids = {ch.doc_id for ch in chunk_map.values()}
    retrieved_chunks: list[Chunk] = [
        chunk_map[rc.chunk_id]
        for rc in record.retrieved_chunks
        if rc.chunk_id in chunk_map
    ]
    tool_texts: list[str] = [
        json.dumps(tc.result, ensure_ascii=False)
        for tc in record.tool_calls
        if tc.ok and tc.result is not None
    ]
    retrieved_texts = [c.text for c in retrieved_chunks]

    if not out.citations:
        if tool_texts and any(
            tc.tool_name in {"calculator", "current_date", "http_api"}
            for tc in record.tool_calls
            if tc.ok
        ):
            return True, []
        return False, ["Missing citations on non-refused answer."]

    errors: list[str] = []
    for idx, cit in enumerate(out.citations, start=1):
        if cit.doc_id not in doc_ids:
            errors.append(f"Citation {idx}: unknown doc_id '{cit.doc_id}'.")
            continue
        if not verify_quote_in_evidence(cit.quote, retrieved_texts + tool_texts):
            errors.append(
                f"Citation {idx}: quote '{cit.quote[:60]}' not found in retrieved chunks."
            )
            continue
        # Ensure doc_id matches at least one retrieved chunk containing the quote
        matching_retrieved = [
            c
            for c in retrieved_chunks
            if c.doc_id == cit.doc_id and verify_quote_in_evidence(cit.quote, [c.text])
        ]
        if not matching_retrieved and not verify_quote_in_evidence(cit.quote, tool_texts):
            errors.append(
                f"Citation {idx}: quote not found inside retrieved chunks for doc '{cit.doc_id}'."
            )

    return len(errors) == 0, errors


def match_key_fact(fact: str, answer_text: str, *, fuzzy_threshold: float = 0.78) -> bool:
    """Check if a single key_fact is covered by answer_text via canonical/fuzzy matching."""
    canon_fact = canonicalize_text(fact)
    canon_ans = canonicalize_text(answer_text)
    if not canon_fact or not canon_ans:
        return False

    # Require every number in the key fact to also appear in the answer!
    fact_nums = {_canon_number(n) for n in _NUMBER_RE.findall(canon_fact)}
    ans_nums = {_canon_number(n) for n in _NUMBER_RE.findall(canon_ans)}
    if fact_nums and not fact_nums.issubset(ans_nums):
        return False

    if canon_fact in canon_ans:
        return True

    fact_tokens = [
        t
        for t in re.findall(r"[a-z0-9]+", canon_fact)
        if t not in {"a", "an", "the", "at", "in", "on", "of", "to", "is", "are", "for"}
    ]
    ans_tokens = set(re.findall(r"[a-z0-9]+", canon_ans))
    if fact_tokens:
        token_recall = sum(1 for t in fact_tokens if t in ans_tokens) / len(fact_tokens)
        if token_recall >= 0.75:
            return True

    if len(canon_ans) <= len(canon_fact) + 12:
        return difflib.SequenceMatcher(None, canon_fact, canon_ans).ratio() >= fuzzy_threshold

    step = max(1, len(canon_fact) // 6)
    win = int(len(canon_fact) * 1.2)
    for start in range(0, len(canon_ans) - max(1, len(canon_fact) // 2) + 1, step):
        snippet = canon_ans[start : start + win]
        if difflib.SequenceMatcher(None, canon_fact, snippet).ratio() >= fuzzy_threshold:
            return True
    return False


def compute_key_fact_recall(
    key_facts: list[str], answer_text: str
) -> tuple[float, list[str], list[str]]:
    """Return (recall_ratio, matched_facts, missed_facts)."""
    if not key_facts:
        return 1.0, [], []
    matched: list[str] = []
    missed: list[str] = []
    for fact in key_facts:
        if match_key_fact(fact, answer_text):
            matched.append(fact)
        else:
            missed.append(fact)
    recall = round(len(matched) / len(key_facts), 4)
    return recall, matched, missed


def find_ungrounded_numbers(
    answer_text: str,
    retrieved_texts: list[str],
    tool_outputs: list[str],
    question_text: str,
) -> list[str]:
    """Identify numeric values in answer_text absent from retrieved context and tools."""
    canon_ans = canonicalize_text(answer_text)
    ans_numbers = [_canon_number(m) for m in _NUMBER_RE.findall(canon_ans)]
    if not ans_numbers:
        return []

    evidence_blob = canonicalize_text(
        "\n".join([*retrieved_texts, *tool_outputs, question_text])
    )
    allowed_numbers = {_canon_number(m) for m in _NUMBER_RE.findall(evidence_blob)}

    ungrounded: list[str] = []
    for num in ans_numbers:
        if num not in allowed_numbers and num not in ungrounded:
            ungrounded.append(num)
    return ungrounded


def wilson_interval(passes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score 95% confidence interval for small-sample binomial rates."""
    if total <= 0:
        return 0.0, 0.0
    p = passes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Compute Pearson correlation between two numeric sequences, or None if degenerate."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    var_x = sum(a * a for a in dx)
    var_y = sum(b * b for b in dy)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return round(sum(a * b for a, b in zip(dx, dy, strict=True)) / math.sqrt(var_x * var_y), 4)
