"""Build candidate payment plans (full/partial/installments/wait), search for
permitted spending changes when needed, and rank safe candidates."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .events import CashEvent, UserFinancialModel
from .forecast import (
    Forecast,
    amount_safe_to_pay,
    build_forecast,
    earliest_full_payment_date,
    format_amount,
    plan_is_safe,
)

EPS = 1e-6


@dataclass
class Candidate:
    method: str  # full_payment | partial_payment | installments | wait
    status: str  # affordable_now | affordable_with_plan | affordable_later
    payments: List[Tuple[date, float]]
    meets_deadline: bool
    spending_changes: List[str] = field(default_factory=list)
    payment_option_id: Optional[str] = None

    @property
    def total_paid(self) -> float:
        return sum(a for _, a in self.payments)

    @property
    def start_date(self) -> date:
        return self.payments[0][0] if self.payments else date.max

    @property
    def n_payments(self) -> int:
        return len(self.payments)

    def sort_key(self):
        opt_num = 10**9
        if self.payment_option_id:
            try:
                opt_num = int(self.payment_option_id.split("_")[-1])
            except Exception:
                pass
        return (
            0 if self.meets_deadline else 1,
            1 if self.spending_changes else 0,
            round(self.total_paid, 2),
            self.start_date,
            self.n_payments,
            opt_num,
        )


def _installment_months(number_of_payments: int, frequency_days: float) -> float:
    if number_of_payments <= 1:
        return 0.0
    return ((number_of_payments - 1) * frequency_days) / 30.0


def build_installment_candidates(
    profile,
    forecast: Forecast,
    request_date: date,
    desired_completion_date: date,
    payment_options: pd.DataFrame,
    request_id: str,
) -> List[Candidate]:
    out = []
    # A blank max_installment_months means the user will not consider
    # installments at all (see AGENTS.md / README dataset contract), and
    # installments must additionally appear in the user's accepted payment
    # methods. Either condition failing means no installment candidate
    # should ever be built for this user, regardless of what options exist.
    if "installments" not in profile.payment_methods or profile.max_installment_months is None:
        return out
    opts = payment_options[
        (payment_options["request_id"] == request_id)
        & (payment_options["payment_method"] == "installments")
    ]
    for _, opt in opts.iterrows():
        months = _installment_months(int(opt["number_of_payments"]), float(opt["payment_frequency_days"]))
        if months > profile.max_installment_months + 0.5:
            continue
        payments = []
        d = opt["first_payment_date"]
        for i in range(int(opt["number_of_payments"])):
            payments.append((d + timedelta(days=int(opt["payment_frequency_days"]) * i), float(opt["payment_amount"])))
        if not plan_is_safe(profile, forecast, payments, request_date):
            continue
        last_date = payments[-1][0]
        out.append(
            Candidate(
                method="installments",
                status="affordable_with_plan",
                payments=payments,
                meets_deadline=last_date <= desired_completion_date,
                payment_option_id=opt["payment_option_id"],
            )
        )
    return out


def build_base_candidates(
    profile,
    forecast: Forecast,
    request_date: date,
    desired_completion_date: date,
    requested_amount: float,
    allows_partial: bool,
    payment_options: pd.DataFrame,
    request_id: str,
) -> List[Candidate]:
    cands: List[Candidate] = []
    safe_today = amount_safe_to_pay(forecast, requested_amount)
    earliest_full = earliest_full_payment_date(forecast, requested_amount)

    if "full_payment" in profile.payment_methods and safe_today >= requested_amount - EPS:
        cands.append(
            Candidate(
                method="full_payment",
                status="affordable_now",
                payments=[(request_date, requested_amount)],
                meets_deadline=request_date <= desired_completion_date,
            )
        )

    if (
        allows_partial
        and "partial_payment" in profile.payment_methods
        and 0 < safe_today < requested_amount - EPS
        and earliest_full is not None
        and earliest_full <= desired_completion_date
    ):
        remainder = requested_amount - safe_today
        cands.append(
            Candidate(
                method="partial_payment",
                status="affordable_with_plan",
                payments=[(request_date, round(safe_today, 2)), (earliest_full, round(remainder, 2))],
                meets_deadline=True,
            )
        )

    cands.extend(
        build_installment_candidates(
            profile, forecast, request_date, desired_completion_date, payment_options, request_id
        )
    )

    if "full_payment" in profile.payment_methods and earliest_full is not None and earliest_full > request_date:
        cands.append(
            Candidate(
                method="wait",
                status="affordable_later",
                payments=[(earliest_full, requested_amount)],
                meets_deadline=earliest_full <= desired_completion_date,
            )
        )
    return cands


def _flex_change_candidates(model: UserFinancialModel, events: List[CashEvent]) -> List[Tuple[str, str, float, float]]:
    """(event_id, action, total_gain, new_amount_if_reduce) for eligible
    flexible spending. Both 'actual' events (a real future row) and
    'projected' recurring occurrences (which carry their anchor's real
    event_id - see events.py) are considered: since several future
    occurrences of a recurring series share one anchor event_id, "stop" or
    "reduce_to" naturally applies to *all* of them as a durable behavior
    change, and the reported gain is summed across every occurrence within
    the forecast horizon."""
    by_id: Dict[str, List[CashEvent]] = {}
    for e in events:
        by_id.setdefault(e.event_id, []).append(e)

    out = []
    for event_id, occurrences in by_id.items():
        rep = occurrences[0]
        can_stop = model.is_eligible_to_stop(rep.category, rep.flexibility)
        can_reduce = model.is_eligible_to_reduce(rep.category, rep.flexibility)
        stop_gain = sum(o.amount for o in occurrences) if can_stop else -1
        reduce_gain = -1
        new_amount = None
        if can_reduce and rep.minimum_allowed_amount is not None:
            reduce_gain = sum(max(0.0, o.amount - rep.minimum_allowed_amount) for o in occurrences)
            new_amount = rep.minimum_allowed_amount
        if stop_gain < 0 and reduce_gain < 0:
            continue
        if stop_gain >= reduce_gain:
            out.append((event_id, "stop", stop_gain, 0.0))
        else:
            out.append((event_id, "reduce", reduce_gain, new_amount))
    out.sort(key=lambda t: -t[2])
    return out


def try_with_spending_changes(
    profile,
    model: UserFinancialModel,
    base_events: List[CashEvent],
    request_date: date,
    desired_completion_date: date,
    requested_amount: float,
    allows_partial: bool,
    payment_options: pd.DataFrame,
    request_id: str,
    max_changes: int = 3,
) -> Optional[Tuple[Candidate, List[str]]]:
    """Search for a permitted combination of stop/reduce spending changes
    that makes some accepted payment method meet the deadline.

    A single greedy pass (always add the single biggest-gain change first)
    can pick a change that raises *total* slack the most while missing the
    one that actually relieves the *nearest* binding constraint - e.g.
    reducing a large but late-occurring expense doesn't help if the tightest
    point in the forecast is next week, while a smaller near-term expense
    would. So instead we explore every combination of up to `max_changes`
    eligible changes (bounded to the top few by individual gain to keep this
    cheap) and rank every combination that succeeds using the same six
    tie-break rules as `build_base_candidates`, rather than accepting the
    first combination that happens to clear the deadline.
    """
    flex = _flex_change_candidates(model, base_events)
    flex = flex[:8]  # bound combinatorics; these are already sorted by gain

    def apply_subset(subset) -> List[CashEvent]:
        events = [CashEvent(**vars(e)) for e in base_events]
        stop_ids = {eid for eid, action, _, _ in subset if action == "stop"}
        reduce_map = {eid: new_amt for eid, action, _, new_amt in subset if action == "reduce"}
        events = [e for e in events if e.event_id not in stop_ids]
        for e in events:
            if e.event_id in reduce_map:
                e.amount = reduce_map[e.event_id]
        return events

    best: Optional[Tuple[Candidate, List[str]]] = None
    for size in range(1, min(max_changes, len(flex)) + 1):
        for subset in itertools.combinations(flex, size):
            events = apply_subset(subset)
            forecast = build_forecast(profile, events, request_date)
            cands = build_base_candidates(
                profile,
                forecast,
                request_date,
                desired_completion_date,
                requested_amount,
                allows_partial,
                payment_options,
                request_id,
            )
            # Spending changes may only steer an *active* payment plan
            # (pay in full now, or follow a fixed installment schedule) -
            # both `amount_safe_to_pay` and `earliest_date_for_full_payment`
            # are pinned by the problem rules to the *baseline* (no-change)
            # forecast, so "wait" (a passive, no-action method) and
            # "partial_payment" (whose two payments are defined *as*
            # amount_safe_to_pay and earliest_date_for_full_payment) can
            # never legitimately be built from a changes-adjusted forecast:
            # doing so silently produces a plan whose dates/amounts don't
            # match the baseline values the output actually reports for
            # those two fields, and for "wait" in particular masks a
            # not-really-passive plan as one requiring no active choice.
            cands = [c for c in cands if c.method in ("full_payment", "installments")]
            deadline_ok = [c for c in cands if c.meets_deadline]
            if not deadline_ok:
                continue
            deadline_ok.sort(key=lambda c: c.sort_key())
            candidate = deadline_ok[0]
            applied = sorted(
                (
                    f"stop:{eid}" if action == "stop" else f"reduce_to:{eid}:{format_amount(new_amt)}"
                    for eid, action, _, new_amt in subset
                )
            )
            candidate.spending_changes = applied
            key = candidate.sort_key() + (len(subset),)
            if best is None or key < (best[0].sort_key() + (len(best[1]),)):
                best = (candidate, applied)
        if best is not None:
            # A solution exists at this size; per the ranking rules, "fewer
            # spending changes" isn't an explicit tie-break beyond the
            # binary "no changes" check, but among otherwise-tied plans a
            # smaller change set is the more conservative pick, so we stop
            # growing the search once we have any working solution.
            break
    return best
