"""
Extract structured financial signals from `messages.csv` text.

IMPORTANT SAFETY NOTE
----------------------
Message and image content is untrusted evidence about the user's finances.
We only ever pull out a fixed set of typed facts (amount / date / currency /
category signal). We never execute or obey any instruction embedded in a
message (e.g. a scam message that says "pay a release charge to receive your
prize" must never cause us to create a payment or income event). The parser
below is a closed-vocabulary classifier: if a message doesn't match a known
template family, it is ignored rather than trusted at face value.

The dataset's messages are heavily templated in English and Indonesian
(bilingual synthetic data). We match on stable template fragments rather than
free-form language understanding, which is precise, cheap (no LLM calls
needed), and auditable. An optional LLM fallback (see llm.py) can be enabled
for messages that don't match any known template, when ANTHROPIC_API_KEY is
set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

CUR_RE = r"(INR|ZAR|IDR|USD|EUR)"
NUM_RE = r"([\d][\d,\.]*)"
DATE_RE = r"(\d{4}-\d{2}-\d{2})"


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


@dataclass
class Signal:
    kind: str
    amount: Optional[float] = None
    currency: Optional[str] = None
    effective_date: Optional[date] = None
    pct: Optional[float] = None
    note: str = ""


# Each rule: (kind, compiled regex, extractor fn(match) -> Signal)
def _mk(kind, pattern, fn):
    return (kind, re.compile(pattern, re.IGNORECASE | re.DOTALL), fn)


RULES = [
    # Salary increased to X, effective from DATE
    _mk(
        "salary_increase",
        rf"(?:increased to|naik menjadi)\s*{CUR_RE}\s*{NUM_RE}.*?(?:applies from|berlaku mulai)\s*{DATE_RE}",
        lambda m: Signal("salary_increase", _num(m.group(2)), m.group(1), _date(m.group(3))),
    ),
    # Salary reduced to X due to unpaid leave ("next salary is reduced to")
    _mk(
        "salary_reduce_temp",
        rf"(?:reduced to|dikurangi menjadi)\s*{CUR_RE}\s*{NUM_RE}",
        lambda m: Signal("salary_reduce_temp", _num(m.group(2)), m.group(1)),
    ),
    # Temporary monthly pay is X (reduced, continues for next payroll)
    _mk(
        "salary_temp_pay",
        rf"temporary monthly pay is\s*{CUR_RE}\s*{NUM_RE}|gaji bulanan sementara.*?{CUR_RE}\s*{NUM_RE}",
        lambda m: Signal(
            "salary_temp_pay",
            _num(m.group(2) or m.group(4)),
            m.group(1) or m.group(3),
        ),
    ),
    # Household employment record ended -> remaining confirmed monthly salary is X
    _mk(
        "salary_partial_end",
        rf"(?:household employment record has ended|sumber pendapatan kerja rumah tangga telah berakhir).*?"
        rf"(?:remaining confirmed monthly salary is|sisa gaji bulanan yang dikonfirmasi adalah)\s*{CUR_RE}\s*{NUM_RE}",
        lambda m: Signal("salary_partial_end", _num(m.group(2)), m.group(1)),
    ),
    # Employment has ended. No regular salary payments scheduled after final settlement.
    _mk(
        "salary_ended",
        r"employment has ended|hubungan kerja anda telah berakhir",
        lambda m: Signal("salary_ended"),
    ),
    # Seasonal contract ended, no renewal confirmed -> income stream stops
    _mk(
        "salary_ended",
        r"seasonal contract.*?has ended|kontrak musiman.*?telah berakhir",
        lambda m: Signal("salary_ended"),
    ),
    # First salary will be X, confirmed credit date DATE (new job)
    _mk(
        "salary_new_first",
        rf"first salary (?:will be|of)\s*{CUR_RE}\s*{NUM_RE}.*?(?:confirmed credit date is|scheduled for|confirmed for|it is confirmed for)\s*{DATE_RE}"
        rf"|gaji pertama.*?sebesar\s*{CUR_RE}\s*{NUM_RE}.*?(?:dijadwalkan pada|dikonfirmasi untuk)\s*{DATE_RE}",
        lambda m: Signal(
            "salary_new_first",
            _num(m.group(2) or m.group(5)),
            m.group(1) or m.group(4),
            _date(m.group(3) or m.group(6)),
        ),
    ),
    # Salary of X is confirmed for DATE (fx settlement note)
    _mk(
        "salary_confirmed_amount_date",
        rf"salary of\s*{CUR_RE}\s*{NUM_RE} is confirmed for\s*{DATE_RE}"
        rf"|gaji sebesar\s*{CUR_RE}\s*{NUM_RE} dikonfirmasi untuk\s*{DATE_RE}",
        lambda m: Signal(
            "salary_confirmed_amount_date",
            _num(m.group(2) or m.group(5)),
            m.group(1) or m.group(4),
            _date(m.group(3) or m.group(6)),
        ),
    ),
    # Confirmed salary is now expected on DATE (date amendment only)
    _mk(
        "salary_date_change",
        rf"confirmed salary is now expected on\s*{DATE_RE}"
        rf"|gaji yang sudah dikonfirmasi kini diperkirakan masuk pada\s*{DATE_RE}",
        lambda m: Signal("salary_date_change", effective_date=_date(m.group(1) or m.group(2))),
    ),
    # Regular salary resumes on DATE (+ new recurring childcare payment - informational only)
    _mk(
        "salary_resume",
        rf"regular salary of\s*{CUR_RE}\s*{NUM_RE} resumes on\s*{DATE_RE}"
        rf"|gaji rutin.*?{CUR_RE}\s*{NUM_RE}.*?(?:resumes|dimulai kembali)?.*?{DATE_RE}",
        lambda m: Signal(
            "salary_resume",
            _num(m.group(2) or m.group(5)) if m.group(2) or m.group(5) else None,
            m.group(1) or m.group(4),
            _date(m.group(3) or m.group(6)) if (m.group(3) or m.group(6)) else None,
        ),
    ),
    # Confirmed base salary is X (commission still pending -> exclude commission, no amount change to base)
    _mk(
        "salary_base_confirm",
        rf"confirmed base salary is\s*{CUR_RE}\s*{NUM_RE}|gaji pokok yang dikonfirmasi adalah\s*{CUR_RE}\s*{NUM_RE}",
        lambda m: Signal("salary_base_confirm", _num(m.group(2) or m.group(4)), m.group(1) or m.group(3)),
    ),
    # Rent increases by X%
    _mk(
        "rent_increase_pct",
        r"increases monthly rent by\s*(\d+)%|menaikkan biaya sewa bulanan sebesar\s*(\d+)%",
        lambda m: Signal("rent_increase_pct", pct=float(m.group(1) or m.group(2))),
    ),
    # Investment sale proceeds settled -> countable settled cash (informational confirm)
    _mk(
        "investment_sale_settled",
        r"proceeds from your investment sale have settled|hasil penjualan investasi anda sudah masuk",
        lambda m: Signal("investment_sale_settled"),
    ),
    # Unrealized / market value moved, no cash -> exclude
    _mk(
        "unrealized_exclude",
        r"no units have been sold and no cash proceeds|belum dijual dan tidak ada transaksi tunai|has not been sold and there has been no cash",
        lambda m: Signal("unrealized_exclude"),
    ),
    # Gig payout still pending / weekly earnings can change -> exclude from countable cash
    _mk(
        "pending_exclude",
        r"payout is still pending|pembayaran berikutnya.*?masih tertunda|still in payment processing|masih dalam proses pembayaran"
        r"|refund has been initiated but has not reached|pengembalian dana sudah diproses, tetapi belum masuk"
        r"|foreign-currency refund is still processing|pengembalian dana dalam mata uang asing.*?masih diproses"
        r"|commission shown for open deals is still pending|komisi dari transaksi yang masih berjalan belum disetujui"
        r"|quarterly bonus is still subject to the final performance review|bonus kuartalan anda masih menunggu",
        lambda m: Signal("pending_exclude"),
    ),
    # Internal transfer between own accounts -> not real spending/income, ignore both legs
    _mk(
        "internal_transfer_exclude",
        r"matching debit and credit came from a transfer between your two accounts"
        r"|debit dan kredit dengan jumlah yang sama berasal dari transfer antara dua rekening",
        lambda m: Signal("internal_transfer_exclude"),
    ),
    # Prize / lottery scam bait ("pay a fee to receive your prize") -> never actionable
    _mk(
        "scam_ignore",
        r"pay the release charge|pay the processing charge|bayar biaya pencairan|bayar biaya pemrosesan",
        lambda m: Signal("scam_ignore"),
    ),
    # Prize proceeds reached account (settled, one-time, closed, no more payments)
    _mk(
        "one_time_settled_credit",
        r"prize proceeds have reached your account after withholding|hasil hadiah telah masuk ke rekening"
        r"|reimbursement for your earlier work expense.*?claim is now closed",
        lambda m: Signal("one_time_settled_credit"),
    ),
    # Invoice payment approved for X, settlement expected DATE
    _mk(
        "invoice_confirmed",
        rf"client approved an invoice payment of\s*{CUR_RE}\s*{NUM_RE}.*?(?:settlement is expected on|penyelesaian diperkirakan pada)\s*{DATE_RE}"
        rf"|klien menyetujui pembayaran faktur sebesar\s*{CUR_RE}\s*{NUM_RE}.*?{DATE_RE}",
        lambda m: Signal(
            "invoice_confirmed",
            _num(m.group(2) or m.group(5)),
            m.group(1) or m.group(4),
            _date(m.group(3) or m.group(6)),
        ),
    ),
    # Debit attempt failed, bill still outstanding, will be retried -> keep as still-due
    _mk(
        "failed_retry_note",
        r"previous debit attempt failed|debit attempt (?:has)? failed|percobaan debit sebelumnya gagal",
        lambda m: Signal("failed_retry_note"),
    ),
    # Card dispute open, no reversal posted -> do not add a credit back
    _mk(
        "dispute_no_reversal",
        r"reversal has not been posted|dana pembalikannya belum tercatat",
        lambda m: Signal("dispute_no_reversal"),
    ),
    # Foreign currency amount only confirmed at settlement -> use existing (settled) amount as-is, no override
    _mk(
        "fx_settle_uncertain",
        r"bank will confirm the final home-currency amount when the transaction settles"
        r"|bank anda akan mengonfirmasi jumlah akhir dalam mata uang utama saat transaksi selesai",
        lambda m: Signal("fx_settle_uncertain"),
    ),
]


def extract_signals(message_text: str) -> List[Signal]:
    """Run all known templates against a message; return every match.

    Order matters only in that more specific patterns are listed first so a
    message rarely double-fires on unrelated categories; duplicates are
    harmless since callers de-duplicate by (kind, event linkage).
    """
    if not isinstance(message_text, str) or not message_text.strip():
        return []
    out: List[Signal] = []
    for kind, pattern, fn in RULES:
        m = pattern.search(message_text)
        if m:
            try:
                sig = fn(m)
                out.append(sig)
            except Exception:
                continue
    return out
