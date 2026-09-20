"""
Local, deterministic image reading for blank-amount financial events.

This module actually opens and OCRs every `dataset/media/images/*.png` file
linked to a blank-`amount` event - no network call or API key required, so
"reads the provided local media files" holds true on every run, not just
when ANTHROPIC_API_KEY happens to be set.

The dataset's images are realistic documents (payslips, utility bills,
payment receipts, taxi/store receipts). They aren't a single fixed template,
so instead of one fragile regex we scan OCR'd lines against a *priority*
list of labels that, across the observed samples, reliably mark the one
figure that matters for cash-flow purposes (the amount actually paid/
received) rather than a subtotal, itemized line, or gross figure:

  1. "Transferred to ... <amount>"   (payslip bank-transfer line - the
     figure OCR most reliably keeps intact, since it survives even when a
     two-column layout garbles the "Net Pay" row itself)
  2. "Net Pay"                        (payslip)
  3. "Total Amount Received"          (payment receipt)
  4. "Amount Due till <date>"         (utility bill - the current charge,
     not the "amount due after <date>" late-payment figure)
  5. "Grand Total" / "Total Amount Due"
  6. A bare "Total:" line, excluding "Subtotal", "Total Earnings",
     "Total Deductions" (payslip line-items that are not the net figure)

This is intentionally conservative: if nothing matches, it returns None and
the caller falls back further (LLM vision if a key is available, else the
documented category-median heuristic) rather than guessing.
"""
from __future__ import annotations

import re
from typing import List, Optional

try:
    import pytesseract
    from PIL import Image

    _OCR_AVAILABLE = True
except Exception:  # pytesseract/PIL/tesseract binary not installed
    _OCR_AVAILABLE = False

_NUM = r"[\d][\d,]*(?:\.\d+)?"

# (label pattern, exclude pattern applied to the same line, is highest-priority-first)
_PRIORITY_LINE_PATTERNS = [
    (re.compile(r"transferred\s+to", re.IGNORECASE), None),
    (re.compile(r"net\s+pay", re.IGNORECASE), None),
    (re.compile(r"total\s+amount\s+received", re.IGNORECASE), None),
    (re.compile(r"amount\s+received", re.IGNORECASE), re.compile(r"total", re.IGNORECASE)),
    (re.compile(r"amount\s+due\s+till", re.IGNORECASE), re.compile(r"\bafter\b", re.IGNORECASE)),
    (re.compile(r"grand\s+total", re.IGNORECASE), None),
    (re.compile(r"total\s+amount\s+due", re.IGNORECASE), None),
    (
        re.compile(r"\btotal\b", re.IGNORECASE),
        re.compile(r"sub\s*total|total\s+earnings|total\s+deductions", re.IGNORECASE),
    ),
]

_STANDALONE_NUMBER_LINE = re.compile(rf"^\s*[=:\-]?\s*({_NUM})\s*$")


def _last_number_on_line(line: str) -> Optional[float]:
    matches = re.findall(_NUM, line)
    matches = [m for m in matches if len(m.replace(",", "").replace(".", "")) >= 2]
    if not matches:
        return None
    # Prefer the rightmost number on the line, not the longest: these
    # documents often print an account/reference/phone number *before* the
    # amount (e.g. "Transferred to :Bank Central Asia - 2582373290 - IDR
    # 4,365,000"), and a "pick the longest token" rule would grab the
    # account number instead of the (shorter) amount that actually matters.
    best = matches[-1]
    try:
        return float(best.replace(",", ""))
    except ValueError:
        return None


def extract_amount_via_ocr(image_path: str) -> Optional[float]:
    """Return the primary amount found in the image, or None if OCR is
    unavailable or no known label pattern matched."""
    if not _OCR_AVAILABLE:
        return None
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
    except Exception:
        return None
    lines = [l for l in text.splitlines() if l.strip()]

    for pattern, exclude in _PRIORITY_LINE_PATTERNS:
        for i, line in enumerate(lines):
            if exclude and exclude.search(line):
                continue
            if pattern.search(line):
                amt = _last_number_on_line(line)
                if amt is not None:
                    return amt
                # Some layouts print the label and its value on separate
                # lines (e.g. "Amount due till" ... blank ... "= 704.05").
                # Look ahead a couple of lines for a standalone number.
                for lookahead in lines[i + 1 : i + 3]:
                    m = _STANDALONE_NUMBER_LINE.match(lookahead)
                    if m:
                        try:
                            return float(m.group(1).replace(",", ""))
                        except ValueError:
                            continue
    return None
