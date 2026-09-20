"""Orchestrate: build forecast -> candidates -> (spending changes if needed)
-> pick winner -> format the final output.csv row for one request."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from .events import UserFinancialModel
from .forecast import amount_safe_to_pay, build_forecast, earliest_full_payment_date, format_amount
from .plans import Candidate, build_base_candidates, try_with_spending_changes


def _fmt_plan(payments) -> str:
    if not payments:
        return "none"
    return "|".join(
        f"{d.isoformat()}:{format_amount(a)}" for d, a in sorted(payments, key=lambda p: p[0])
    )


def _fmt_changes(changes) -> str:
    if not changes:
        return "none"
    return "|".join(changes[:3])


def decide(
    request_row: pd.Series,
    model: UserFinancialModel,
    payment_options: pd.DataFrame,
) -> dict:
    request_id = request_row["request_id"]
    request_date: date = request_row["request_date"]
    desired_completion_date: date = request_row["desired_completion_date"]
    requested_amount = float(request_row["requested_amount"])
    allows_partial = bool(request_row["allows_partial_payment"]) if not pd.isna(
        request_row["allows_partial_payment"]
    ) else False

    events = model.all_cash_events(request_date)
    forecast = build_forecast(model.profile, events, request_date)

    base_safe = amount_safe_to_pay(forecast, requested_amount)
    base_earliest = earliest_full_payment_date(forecast, requested_amount)

    candidates = build_base_candidates(
        model.profile,
        forecast,
        request_date,
        desired_completion_date,
        requested_amount,
        allows_partial,
        payment_options,
        request_id,
    )
    deadline_ok = [c for c in candidates if c.meets_deadline]

    winner: Optional[Candidate] = None
    used_changes: list = []
    if deadline_ok:
        deadline_ok.sort(key=lambda c: c.sort_key())
        winner = deadline_ok[0]
    else:
        result = try_with_spending_changes(
            model.profile,
            model,
            events,
            request_date,
            desired_completion_date,
            requested_amount,
            allows_partial,
            payment_options,
            request_id,
        )
        if result:
            winner, used_changes = result
        elif candidates:
            candidates.sort(key=lambda c: c.sort_key())
            winner = candidates[0]  # best available even if late (affordable_later)

    if winner is None:
        return {
            "request_id": request_id,
            "amount_safe_to_pay": round(base_safe, 2),
            "affordability_status": "not_affordable",
            "recommended_payment_method": "not_recommended",
            "payment_plan": "none",
            "earliest_date_for_full_payment": base_earliest.isoformat() if base_earliest else "",
            "spending_changes_needed": "none",
            "decision_explanation": (
                f"No eligible payment method from the user's accepted list is safe within the "
                f"90-day forecast without breaking the {model.profile.minimum_balance:,.2f} "
                f"{model.profile.home_currency} minimum balance."
            ),
        }

    explanation = _explain(winner, model, base_safe, requested_amount)
    return {
        "request_id": request_id,
        "amount_safe_to_pay": round(base_safe, 2),
        "affordability_status": winner.status,
        "recommended_payment_method": winner.method,
        "payment_plan": _fmt_plan(winner.payments),
        "earliest_date_for_full_payment": (
            base_earliest.isoformat() if base_earliest else ""
        ) if winner.status != "affordable_now" else request_date.isoformat(),
        "spending_changes_needed": _fmt_changes(winner.spending_changes),
        "decision_explanation": explanation,
    }


def _explain(winner: Candidate, model, base_safe: float, requested_amount: float) -> str:
    cur = model.profile.home_currency
    if winner.method == "full_payment":
        return (
            f"Pay {cur} {requested_amount:,.2f} on {winner.payments[0][0].isoformat()}. "
            f"This keeps the balance at or above the {cur} {model.profile.minimum_balance:,.2f} "
            f"minimum through the 90-day forecast."
        )
    if winner.method == "partial_payment":
        p1, p2 = winner.payments
        return (
            f"Pay {cur} {p1[1]:,.2f} now and the remaining {cur} {p2[1]:,.2f} on "
            f"{p2[0].isoformat()}, once projected cash flow safely covers it."
        )
    if winner.method == "installments":
        return (
            f"Use the {len(winner.payments)}-payment installment option "
            f"({winner.payment_option_id}) starting {winner.payments[0][0].isoformat()}; "
            f"each payment keeps the balance above the {cur} {model.profile.minimum_balance:,.2f} minimum."
        )
    if winner.method == "wait":
        return (
            f"Wait and pay the full {cur} {requested_amount:,.2f} on "
            f"{winner.payments[0][0].isoformat()}, the earliest date the forecast shows it "
            f"as safe; paying sooner would breach the minimum balance."
        )
    return "No safe payment path was found within the forecast window."
