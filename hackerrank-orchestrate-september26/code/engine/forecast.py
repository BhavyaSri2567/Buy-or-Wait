"""90-day balance forecasting and the core safety-check math."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from .events import CashEvent, HORIZON_DAYS, Profile


@dataclass
class Forecast:
    dates: List[date]
    balance: List[float]  # end-of-day balance, before any request payment
    slack: List[float]  # balance - minimum_balance
    suffix_min_slack: List[float]  # min(slack[t:]) for t..end


def build_forecast(profile: Profile, events: List[CashEvent], request_date: date) -> Forecast:
    n = HORIZON_DAYS + 1
    dates = [request_date + timedelta(days=i) for i in range(n)]
    net_flow = [0.0] * n
    for e in events:
        idx = (e.on_date - request_date).days
        if 0 <= idx < n:
            signed = e.amount if e.direction == "credit" else -e.amount
            net_flow[idx] += signed

    balance = [0.0] * n
    running = profile.balance
    for i in range(n):
        running += net_flow[i]
        balance[i] = running

    slack = [b - profile.minimum_balance for b in balance]
    suffix_min = [0.0] * n
    running_min = float("inf")
    for i in range(n - 1, -1, -1):
        running_min = min(running_min, slack[i])
        suffix_min[i] = running_min

    return Forecast(dates=dates, balance=balance, slack=slack, suffix_min_slack=suffix_min)


def format_amount(a: float) -> str:
    """Render a plan/spending-change amount the way the dataset's gold
    answers do: a bare integer when the value has no cents (e.g. "25256",
    never "25256.0"), otherwise a fixed two-decimal string (e.g. "620.40",
    never "620.4"). Plain Python float interpolation (`f"{round(a,2)}"`)
    does neither - it prints "25256.0" for whole numbers and drops
    trailing zero cents - which breaks the exact-string comparisons used
    for `payment_plan` and `spending_changes_needed`."""
    r = round(a, 2)
    if abs(r - round(r)) < 1e-9:
        return str(int(round(r)))
    return f"{r:.2f}"


def amount_safe_to_pay(forecast: Forecast, requested_amount: float) -> float:
    slack_now = forecast.suffix_min_slack[0]
    return max(0.0, min(requested_amount, slack_now))


def earliest_full_payment_date(forecast: Forecast, requested_amount: float) -> Optional[date]:
    for i, s in enumerate(forecast.suffix_min_slack):
        if s >= requested_amount - 1e-6:
            return forecast.dates[i]
    return None


def plan_is_safe(
    profile: Profile,
    forecast: Forecast,
    payments: List[Tuple[date, float]],
    request_date: date,
) -> bool:
    """Overlay extra one-off debits (the plan's payments) on the baseline
    forecast and confirm the balance never dips below minimum, for any
    payment date that falls inside the 90-day horizon."""
    n = len(forecast.dates)
    extra = [0.0] * n
    for pay_date, amt in payments:
        idx = (pay_date - request_date).days
        if idx < 0:
            return False
        if idx < n:
            extra[idx] -= amt
    cum_extra = 0.0
    for i in range(n):
        cum_extra += extra[i]
        if forecast.balance[i] + cum_extra < profile.minimum_balance - 1e-6:
            return False
    return True
