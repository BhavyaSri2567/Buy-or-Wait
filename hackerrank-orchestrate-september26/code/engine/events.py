"""
Reconstruct each user's financial state: a resolved ledger of cash events
plus detected recurring series, ready for 90-day forecasting.

Design (documented simplifications live in the repo README):

1. Every financial_events.csv row is classified as counted / excluded for
   cash-flow purposes based on `status` and `direction`:
     - direction == non_cash                    -> excluded (investment marks)
     - status in {cancelled, failed}             -> excluded
     - status == unrealized                      -> excluded
     - status == pending AND direction == credit -> excluded (not confirmed)
     - status == pending AND direction == debit  -> counted (reserved)
     - status in {settled, scheduled}             -> counted
   Exact duplicate rows (same user/category/direction/amount/date) are
   collapsed to one.

2. Blank `amount` values are resolved via the linked image (LLM vision, when
   ANTHROPIC_API_KEY is set) or, if unavailable, a conservative fallback:
   debits default to the user's historical median for that category (a
   grounded estimate, not an invented one); credits are excluded entirely
   until confirmed, per "do not invent unsupported income".

3. Recurring series are detected per (user, category) from *settled*
   historical events at or before `request_date`: with >=3 occurrences and a
   coefficient of variation on the inter-event interval below 0.6, the
   category is treated as recurring and projected forward at the median
   interval and latest (post-amendment) amount through the 90-day horizon.

4. Messages amend the reconstructed picture: an event-linked signal patches
   that specific event; a category-level signal (e.g. "salary increased to
   X from date D", "rent increases by X%", "employment has ended") adjusts
   the projected recurring series going forward.
"""
from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .extraction import Signal, extract_signals
from .fx import FxTable
from .llm import LLMClient
from .ocr import extract_amount_via_ocr

HORIZON_DAYS = 90


@dataclass
class CashEvent:
    event_id: str
    category: str
    direction: str  # debit | credit
    amount: float  # home currency
    on_date: date
    flexibility: str
    minimum_allowed_amount: Optional[float]
    source: str  # "actual" | "projected"
    note: str = ""


@dataclass
class Profile:
    user_id: str
    home_currency: str
    balance: float
    minimum_balance: float
    priorities: List[str]
    protect: List[str]
    willing_reduce: List[str]
    willing_stop: List[str]
    payment_methods: List[str]
    max_installment_months: Optional[float]


def _split(s) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


def build_profile(row: pd.Series) -> Profile:
    return Profile(
        user_id=row["user_id"],
        home_currency=row["home_currency"],
        balance=float(row["current_available_balance"]),
        minimum_balance=float(row["minimum_balance_to_keep"]),
        priorities=_split(row.get("financial_priorities")),
        protect=_split(row.get("expense_categories_to_protect")),
        willing_reduce=_split(row.get("expense_categories_user_is_willing_to_reduce")),
        willing_stop=_split(row.get("expense_categories_user_is_willing_to_stop")),
        payment_methods=_split(row.get("payment_methods_user_will_consider")),
        max_installment_months=(
            float(row["max_installment_months"])
            if pd.notna(row.get("max_installment_months"))
            else None
        ),
    )


class UserFinancialModel:
    """Builds and holds the resolved cash-flow picture for one user."""

    def __init__(
        self,
        profile: Profile,
        events_df: pd.DataFrame,
        images_df: pd.DataFrame,
        messages_df: pd.DataFrame,
        fx: FxTable,
        llm: LLMClient,
        dataset_dir: str,
    ):
        self.profile = profile
        self.fx = fx
        self.llm = llm
        self.dataset_dir = dataset_dir
        self.events_df = events_df[events_df["user_id"] == profile.user_id].copy()
        self.images_df = images_df[images_df["user_id"] == profile.user_id]
        self.messages_df = messages_df[messages_df["user_id"] == profile.user_id]
        self.notes: List[str] = []
        self._resolve_amounts()
        self._dedupe()
        self.signals_by_event: Dict[str, List[Signal]] = {}
        self.category_signals: List[Tuple[Signal, Optional[date]]] = []
        self._collect_message_signals()

    # ---------- amount resolution ----------
    def _resolve_amounts(self):
        blanks = self.events_df[self.events_df["amount"].isna()]
        for idx, row in blanks.iterrows():
            img_row = self.images_df[self.images_df["related_event_id"] == row["event_id"]]
            resolved = None
            method = None
            if not img_row.empty:
                image_id = img_row.iloc[0]["image_id"]
                image_path = os.path.join(
                    self.dataset_dir, "media", "images", f"{image_id}.png"
                )
                # Every blank-amount event with a linked image is resolved by
                # actually opening and reading that local file - via the
                # Anthropic vision API when a key is available (semantic
                # understanding of the document), and always via local OCR
                # (engine/ocr.py, no network/API key required) as a
                # deterministic fallback, so image reading never silently
                # depends on an optional credential.
                result = self.llm.extract_image_amount(image_path)
                if result and result.get("amount") is not None:
                    resolved = float(result["amount"])
                    method = "llm_vision"
                else:
                    ocr_amount = extract_amount_via_ocr(image_path)
                    if ocr_amount is not None:
                        resolved = ocr_amount
                        method = "local_ocr"
                if resolved is not None:
                    self.notes.append(
                        f"{row['event_id']}: blank amount resolved from {image_id} via {method} -> {resolved:.2f}"
                    )
            if resolved is None:
                if row["direction"] == "debit":
                    hist = self.events_df[
                        (self.events_df["category"] == row["category"])
                        & (self.events_df["status"] == "settled")
                        & (self.events_df["amount"].notna())
                    ]["amount"]
                    if len(hist) > 0:
                        resolved = float(hist.median())
                        self.notes.append(
                            f"{row['event_id']}: blank amount, image unresolved; "
                            f"used category median {resolved:.2f} as conservative estimate"
                        )
                    else:
                        resolved = 0.0
                        self.notes.append(
                            f"{row['event_id']}: blank amount unresolved, no history; treated as 0 (excluded)"
                        )
                else:
                    resolved = 0.0
                    self.events_df.loc[idx, "status"] = "pending"
                    self.notes.append(
                        f"{row['event_id']}: blank credit amount unresolved; excluded until confirmed"
                    )
            self.events_df.loc[idx, "amount"] = resolved

    def _dedupe(self):
        subset = ["category", "direction", "amount", "event_date"]
        before = len(self.events_df)
        self.events_df = self.events_df.drop_duplicates(subset=subset, keep="first")
        removed = before - len(self.events_df)
        if removed:
            self.notes.append(f"removed {removed} exact-duplicate event row(s)")

    # ---------- message signal collection ----------
    def _collect_message_signals(self):
        for _, row in self.messages_df.iterrows():
            sigs = extract_signals(row.get("message_text", ""))
            related = row.get("related_event_id")
            sent_at = row.get("sent_at")
            sent_date: Optional[date] = None
            if isinstance(sent_at, str) and sent_at:
                try:
                    sent_date = pd.to_datetime(sent_at).date()
                except Exception:
                    sent_date = None
            for sig in sigs:
                if isinstance(related, str) and related:
                    self.signals_by_event.setdefault(related, []).append(sig)
                else:
                    # Unlinked category-level signals carry their message date
                    # so callers can check whether newer settled data
                    # contradicts them (see FIX #3 in projected_recurring_events).
                    self.category_signals.append((sig, sent_date))

    def signals_for_request(self, request_id: str) -> List[Signal]:
        rows = self.messages_df[self.messages_df["request_id"] == request_id]
        out = []
        for _, row in rows.iterrows():
            out.extend(extract_signals(row.get("message_text", "")))
        return out

    # ---------- counted/excluded classification ----------
    def _counted_rows(self) -> pd.DataFrame:
        df = self.events_df
        df = df[df["direction"] != "non_cash"]
        df = df[~df["status"].isin(["cancelled", "failed", "unrealized"])]
        df = df[~((df["status"] == "pending") & (df["direction"] == "credit"))]
        return df

    def _apply_event_level_signals(self, row: pd.Series) -> Optional[pd.Series]:
        """Returns a possibly-amended copy of the row, or None if excluded."""
        sigs = self.signals_by_event.get(row["event_id"], [])
        row = row.copy()
        for sig in sigs:
            if sig.kind in (
                "pending_exclude",
                "unrealized_exclude",
                "internal_transfer_exclude",
                "dispute_no_reversal",
                "scam_ignore",
            ):
                return None
            if sig.kind in (
                "salary_increase",
                "salary_reduce_temp",
                "salary_temp_pay",
                "salary_partial_end",
                "salary_new_first",
                "salary_confirmed_amount_date",
                "salary_base_confirm",
                "invoice_confirmed",
            ) and sig.amount is not None:
                row["amount"] = sig.amount
                if sig.currency:
                    row["currency"] = sig.currency
            if sig.effective_date is not None and sig.kind in (
                "salary_date_change",
                "salary_new_first",
                "salary_confirmed_amount_date",
                "salary_resume",
                "invoice_confirmed",
            ):
                row["settlement_date"] = sig.effective_date
                row["event_date"] = sig.effective_date
        return row

    def resolved_cash_events(self, horizon_end: date) -> List[CashEvent]:
        """Actual (non-projected) events, in home currency, patched by
        event-linked message signals, within [-inf, horizon_end]."""
        out: List[CashEvent] = []
        for _, row in self._counted_rows().iterrows():
            patched = self._apply_event_level_signals(row)
            if patched is None:
                continue
            on_date = patched["settlement_date"] or patched["event_date"]
            if on_date is None or pd.isna(on_date):
                continue
            if on_date > horizon_end:
                continue
            try:
                amt_home = self.fx.convert(
                    float(patched["amount"]), patched["currency"], self.profile.home_currency, on_date
                )
            except Exception:
                amt_home = float(patched["amount"])  # best effort
            out.append(
                CashEvent(
                    event_id=patched["event_id"],
                    category=patched["category"],
                    direction=patched["direction"],
                    amount=amt_home,
                    on_date=on_date,
                    flexibility=patched.get("flexibility") or "fixed",
                    minimum_allowed_amount=(
                        float(patched["minimum_allowed_amount"])
                        if pd.notna(patched.get("minimum_allowed_amount"))
                        else None
                    ),
                    source="actual",
                )
            )
        return out

    # Categories that are periodic by definition (paid income), where a
    # single confirmed data point already justifies projecting forward -
    # unlike discretionary spending, which needs statistical proof of a
    # pattern. See FIX #1.
    _INHERENTLY_PERIODIC_CATEGORIES = {"salary"}
    _DEFAULT_PERIODIC_INTERVAL_DAYS = 30

    @staticmethod
    def _dominant_amount_cluster(dates: List[date], amounts: List[float], event_ids: List[str]):
        """FIX #2b: some categories (notably 'salary') bundle two genuinely
        different sub-streams under one label - e.g. a stable "Base salary"
        paid every month plus a variable "Performance commission" paid a few
        days later. Averaging gaps/amounts across both conflates them and
        can anchor the projection on the volatile, unpredictable stream.

        If one exact amount value repeats across at least half the
        occurrences (a strong signal of a stable recurring component), keep
        only those occurrences for interval/amount detection and silently
        drop the other stream from the forward projection - projecting a
        variable bonus/commission forward would be inventing income anyway,
        so omitting it is also the financially safer interpretation.
        Returns (dates, amounts, event_ids) - filtered if a dominant cluster
        is found, unchanged otherwise (a no-op for normal variable-but-
        single-stream categories like groceries or dining, where no exact
        value repeats often enough to form a majority).
        """
        if len(amounts) < 4:
            return dates, amounts, event_ids
        counts: Dict[float, int] = {}
        for a in amounts:
            key = round(a, 2)
            counts[key] = counts.get(key, 0) + 1
        best_amount, best_count = max(counts.items(), key=lambda kv: kv[1])
        if best_count >= max(2, len(amounts) // 2) and best_count < len(amounts):
            filtered = [
                (d, a, eid) for d, a, eid in zip(dates, amounts, event_ids) if round(a, 2) == best_amount
            ]
            return [d for d, _, _ in filtered], [a for _, a, _ in filtered], [e for _, _, e in filtered]
        return dates, amounts, event_ids

    @staticmethod
    def _robust_interval_and_anchor(dates: List[date], amounts: List[float], event_ids: List[str]):
        """FIX #2: estimate a recurrence interval and an anchor (date, index)
        that is resistant to a single irregular in-category payment (a bonus,
        arrears adjustment, or short-notice correction). A naive mean() over
        all gaps lets one anomalous gap shift the projected date by weeks;
        the median is far more robust to that single outlier, and picking
        the most recent occurrence whose *own* gap is close to that median
        avoids anchoring the projection on the anomalous row itself.

        Returns (interval_days, anchor_date, anchor_amount, anchor_event_id)
        or None if the series is too irregular to project.
        """
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        gaps = [g for g in gaps if g > 0]
        if len(gaps) < 1:
            return None
        median_gap = statistics.median(gaps)
        if median_gap <= 0:
            return None
        # Coefficient of variation computed against the median (not the
        # mean) so one outlier gap doesn't inflate the "mean" denominator
        # and mask its own distortion.
        mad = statistics.pstdev(gaps)
        cv = mad / median_gap if median_gap else 999
        if cv > 0.6 or median_gap > 100:
            return None  # not clearly periodic within a projectable cadence

        interval = max(1, round(median_gap))
        # Anchor on the most recent occurrence whose preceding gap is within
        # 30% of the median interval, so a one-off bonus/arrears payment
        # (an anomalously short gap) never becomes the projection anchor.
        anchor_idx = len(dates) - 1
        for i in range(len(dates) - 1, 0, -1):
            gap = (dates[i] - dates[i - 1]).days
            if gap > 0 and abs(gap - median_gap) <= 0.3 * median_gap:
                anchor_idx = i
                break
        return interval, dates[anchor_idx], amounts[anchor_idx], event_ids[anchor_idx]

    def projected_recurring_events(
        self, request_date: date, horizon_end: date
    ) -> List[CashEvent]:
        """Detect recurring (user, category) series from settled history at
        or before request_date and project occurrences into the horizon."""
        hist = self._counted_rows()
        hist = hist[hist["status"] == "settled"]
        hist = hist[hist["settlement_date"].notna()]
        hist = hist[hist["settlement_date"] <= request_date]

        out: List[CashEvent] = []
        for (category, direction), grp in hist.groupby(["category", "direction"]):
            grp = grp.sort_values("settlement_date")
            dates = list(grp["settlement_date"])
            amounts = [float(a) for a in grp["amount"]]
            event_ids = list(grp["event_id"])
            dates, amounts, event_ids = self._dominant_amount_cluster(dates, amounts, event_ids)
            is_periodic_by_definition = category in self._INHERENTLY_PERIODIC_CATEGORIES

            # FIX #1: discretionary spending needs >=3 points and >=2 gaps to
            # earn "recurring" status. Income categories that are periodic by
            # definition (salary) only need one confirmed data point - a
            # brand-new hire with a single settled paycheck (or only the
            # "next confirmed salary" row) is still on a monthly cadence; we
            # just can't measure the interval from history, so we assume the
            # standard ~30-day cycle instead of silently dropping all future
            # income.
            if len(dates) < 3 and not is_periodic_by_definition:
                continue

            last_row = grp.iloc[-1]
            last_currency = last_row["currency"]
            flexibility = last_row.get("flexibility") or "fixed"
            min_allowed = (
                float(last_row["minimum_allowed_amount"])
                if pd.notna(last_row.get("minimum_allowed_amount"))
                else None
            )

            if len(dates) >= 2:
                result = self._robust_interval_and_anchor(dates, amounts, event_ids)
                if result is None:
                    if not is_periodic_by_definition:
                        continue
                    interval = self._DEFAULT_PERIODIC_INTERVAL_DAYS
                    last_date, last_amount, anchor_event_id = dates[-1], amounts[-1], event_ids[-1]
                else:
                    interval, last_date, last_amount, anchor_event_id = result
            elif is_periodic_by_definition:
                # Exactly one historical point (e.g. a single "next
                # confirmed salary" row): assume the standard monthly cycle.
                interval = self._DEFAULT_PERIODIC_INTERVAL_DAYS
                last_date, last_amount, anchor_event_id = dates[-1], amounts[-1], event_ids[-1]
            else:
                continue

            # Apply category-level message amendments (salary_*/rent_increase_pct).
            # FIX #3: an unlinked signal (no related_event_id, so we can't
            # confirm it describes this exact row) may be stale. Only let it
            # fully stop a series when there is no *newer* settled
            # occurrence contradicting it; a message claiming income "ended"
            # before the most recent settled paycheck we've actually seen is
            # outdated relative to firmer, newer evidence and is downgraded
            # to informational rather than zeroing real income.
            stop_after = None
            amount_override = last_amount
            currency_override = last_currency
            temp_next_only = None
            for sig, sent_date in self.category_signals:
                contradicted = sent_date is not None and last_date > sent_date
                if category == "salary" and sig.kind == "salary_ended":
                    if not contradicted:
                        stop_after = request_date  # no further occurrences
                elif category == "salary" and sig.kind in ("salary_increase", "salary_partial_end"):
                    if sig.amount is not None and not contradicted:
                        amount_override = sig.amount
                        currency_override = sig.currency or currency_override
                elif category == "salary" and sig.kind in ("salary_reduce_temp", "salary_temp_pay"):
                    if sig.amount is not None and not contradicted:
                        temp_next_only = (sig.amount, sig.currency or currency_override)
                elif category == "rent" and sig.kind == "rent_increase_pct" and sig.pct and not contradicted:
                    amount_override = last_amount * (1 + sig.pct / 100.0)

            if stop_after is not None:
                continue  # income stream ended; no future occurrences projected

            occ_date = last_date + timedelta(days=interval)
            first = True
            n = 0
            while occ_date <= horizon_end and n < 12:
                amt = amount_override
                cur = currency_override
                if first and temp_next_only is not None:
                    amt, cur = temp_next_only
                first = False
                if occ_date >= request_date:
                    try:
                        amt_home = self.fx.convert(amt, cur, self.profile.home_currency, occ_date)
                    except Exception:
                        amt_home = amt
                    out.append(
                        CashEvent(
                            # FIX (spending-change referenceability): use the
                            # anchor's real event_id, not a synthetic one.
                            # spending_changes_needed must cite a real
                            # event_id from financial_events.csv; a future
                            # recurring occurrence has no row of its own, so
                            # we reference the historical event whose pattern
                            # (amount/flexibility/minimum_allowed_amount) it
                            # inherits. This also means "stop"/"reduce_to"
                            # naturally applies to every future occurrence of
                            # that series - a durable behavior change, not a
                            # single skipped payment.
                            event_id=anchor_event_id,
                            category=category,
                            direction=direction,
                            amount=amt_home,
                            on_date=occ_date,
                            flexibility=flexibility,
                            minimum_allowed_amount=min_allowed,
                            source="projected",
                        )
                    )
                occ_date = occ_date + timedelta(days=interval)
                n += 1
        return out

    def all_cash_events(self, request_date: date) -> List[CashEvent]:
        horizon_end = request_date + timedelta(days=HORIZON_DAYS)
        events = self.resolved_cash_events(horizon_end)
        events += self.projected_recurring_events(request_date, horizon_end)
        return [e for e in events if request_date <= e.on_date <= horizon_end]

    def is_protected(self, category: str) -> bool:
        return category in self.profile.protect

    def is_eligible_to_stop(self, category: str, flexibility: str) -> bool:
        return (
            not self.is_protected(category)
            and category in self.profile.willing_stop
            and flexibility in ("stoppable", "reducible_or_stoppable")
        )

    def is_eligible_to_reduce(self, category: str, flexibility: str) -> bool:
        return (
            not self.is_protected(category)
            and category in self.profile.willing_reduce
            and flexibility in ("reducible", "reducible_or_stoppable")
        )
