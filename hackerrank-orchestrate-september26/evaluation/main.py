#!/usr/bin/env python3
"""
Evaluation workflow for the Buy or Wait? solution. Two independent checks:

1. score_against_samples() - runs the live decision engine against
   dataset/sample_requests.csv (25 solved examples, disjoint from
   requests.csv) and reports accuracy on every graded dimension named in
   the challenge: amount_safe_to_pay, affordability_status,
   recommended_payment_method, payment_plan, earliest_date_for_full_payment,
   spending_changes_needed, and basic decision_explanation sanity.

2. validate_output_schema() - a ground-truth-free structural validator that
   runs over the actual output.csv submitted for dataset/requests.csv,
   checking every row against the exact rules in problem_statement.md
   (allowed enum values, 0 <= amount_safe_to_pay <= requested_amount,
   payment_plan format/chronology/arithmetic, spending_changes_needed
   format and stop/reduce mutual exclusivity, date formats). This never
   references sample_requests.csv or any per-request expected answer - it
   only checks internal consistency and schema compliance, so it can be run
   safely on the real submission before uploading.
"""
import os
import re
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

from engine.io_utils import load_dataset
from engine.fx import FxTable
from engine.llm import LLMClient
from engine.events import build_profile, UserFinancialModel
from engine.decision import decide

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(HERE, "dataset")

ALLOWED_STATUS = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
ALLOWED_METHOD = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}
PLAN_ENTRY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}:-?\d+(\.\d+)?$")
CHANGE_ENTRY_RE = re.compile(r"^(stop:[^:|]+|reduce_to:[^:|]+:-?\d+(\.\d+)?)$")


def _build_model_cache(ds, fx, llm):
    profiles_by_user = {row["user_id"]: build_profile(row) for _, row in ds.profiles.iterrows()}
    cache = {}

    def get_model(user_id):
        if user_id not in cache:
            cache[user_id] = UserFinancialModel(
                profile=profiles_by_user[user_id],
                events_df=ds.events,
                images_df=ds.images,
                messages_df=ds.messages,
                fx=fx,
                llm=llm,
                dataset_dir=DATASET_DIR,
            )
        return cache[user_id]

    return get_model, profiles_by_user


def score_against_samples():
    ds = load_dataset(DATASET_DIR)
    fx = FxTable(ds.exchange_rates)
    llm = LLMClient()
    get_model, profiles_by_user = _build_model_cache(ds, fx, llm)

    samp = pd.read_csv(os.path.join(DATASET_DIR, "sample_requests.csv"))
    for col in ("request_date", "desired_completion_date"):
        samp[col] = pd.to_datetime(samp[col]).dt.date

    n = len(samp)
    counters = {
        "status": 0, "method": 0, "payment_plan": 0,
        "earliest_date": 0, "spending_changes": 0,
    }
    amt_abs_err = []
    rows_out = []
    for _, req in samp.iterrows():
        if req["user_id"] not in profiles_by_user:
            continue
        model = get_model(req["user_id"])
        pred = decide(req, model, ds.payment_options)

        counters["status"] += pred["affordability_status"] == req["affordability_status"]
        counters["method"] += pred["recommended_payment_method"] == req["recommended_payment_method"]
        counters["payment_plan"] += pred["payment_plan"] == req["payment_plan"]
        gold_earliest = "" if pd.isna(req["earliest_date_for_full_payment"]) else str(req["earliest_date_for_full_payment"])
        counters["earliest_date"] += pred["earliest_date_for_full_payment"] == gold_earliest
        counters["spending_changes"] += pred["spending_changes_needed"] == req["spending_changes_needed"]
        amt_abs_err.append(abs(pred["amount_safe_to_pay"] - float(req["amount_safe_to_pay"])))

        rows_out.append(
            {
                "request_id": req["request_id"],
                "gold_status": req["affordability_status"], "pred_status": pred["affordability_status"],
                "gold_method": req["recommended_payment_method"], "pred_method": pred["recommended_payment_method"],
                "gold_amount": req["amount_safe_to_pay"], "pred_amount": pred["amount_safe_to_pay"],
                "gold_plan": req["payment_plan"], "pred_plan": pred["payment_plan"],
                "gold_earliest": gold_earliest, "pred_earliest": pred["earliest_date_for_full_payment"],
                "gold_changes": req["spending_changes_needed"], "pred_changes": pred["spending_changes_needed"],
            }
        )

    print(f"=== score_against_samples (n={n}) ===")
    print(f"affordability_status accuracy:        {counters['status']/n:.1%}")
    print(f"recommended_payment_method accuracy:  {counters['method']/n:.1%}")
    print(f"payment_plan exact-match rate:        {counters['payment_plan']/n:.1%}")
    print(f"earliest_date_for_full_payment match: {counters['earliest_date']/n:.1%}")
    print(f"spending_changes_needed exact match:  {counters['spending_changes']/n:.1%}")
    print(f"amount_safe_to_pay mean abs error:    {sum(amt_abs_err)/n:,.2f}")
    pd.DataFrame(rows_out).to_csv(os.path.join(os.path.dirname(__file__), "sample_eval_detail.csv"), index=False)
    print("Per-row detail written to evaluation/sample_eval_detail.csv\n")


def validate_output_schema(output_path=None, requests_path=None):
    """Ground-truth-free structural check of the actual submission file."""
    output_path = output_path or os.path.join(HERE, "output.csv")
    requests_path = requests_path or os.path.join(DATASET_DIR, "requests.csv")

    out = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    requests = pd.read_csv(requests_path)

    errors = []
    expected_cols = [
        "request_id", "amount_safe_to_pay", "affordability_status",
        "recommended_payment_method", "payment_plan",
        "earliest_date_for_full_payment", "spending_changes_needed",
        "decision_explanation",
    ]
    if list(out.columns) != expected_cols:
        errors.append(f"column mismatch: {list(out.columns)}")

    if set(out["request_id"]) != set(requests["request_id"]):
        missing = set(requests["request_id"]) - set(out["request_id"])
        extra = set(out["request_id"]) - set(requests["request_id"])
        if missing:
            errors.append(f"missing {len(missing)} request_id(s), e.g. {list(missing)[:3]}")
        if extra:
            errors.append(f"unexpected {len(extra)} request_id(s), e.g. {list(extra)[:3]}")
    if out["request_id"].duplicated().any():
        errors.append("duplicate request_id rows present")

    req_amount = dict(zip(requests["request_id"], requests["requested_amount"]))
    for _, row in out.iterrows():
        rid = row["request_id"]
        try:
            amt = float(row["amount_safe_to_pay"])
        except ValueError:
            errors.append(f"{rid}: amount_safe_to_pay not numeric: {row['amount_safe_to_pay']!r}")
            continue
        requested = req_amount.get(rid)
        if requested is not None and not (-1e-6 <= amt <= requested + 1e-6):
            errors.append(f"{rid}: amount_safe_to_pay {amt} out of [0, {requested}]")

        if row["affordability_status"] not in ALLOWED_STATUS:
            errors.append(f"{rid}: invalid affordability_status {row['affordability_status']!r}")
        if row["recommended_payment_method"] not in ALLOWED_METHOD:
            errors.append(f"{rid}: invalid recommended_payment_method {row['recommended_payment_method']!r}")

        plan = row["payment_plan"]
        if plan != "none":
            entries = plan.split("|")
            if not all(PLAN_ENTRY_RE.match(e) for e in entries):
                errors.append(f"{rid}: malformed payment_plan {plan!r}")
            else:
                dates = [datetime.strptime(e.split(":")[0], "%Y-%m-%d") for e in entries]
                if dates != sorted(dates):
                    errors.append(f"{rid}: payment_plan not chronological: {plan!r}")

        if row["affordability_status"] == "affordable_now" and row["earliest_date_for_full_payment"]:
            req_date = requests.loc[requests.request_id == rid, "request_date"].iloc[0]
            if row["earliest_date_for_full_payment"] != req_date:
                errors.append(
                    f"{rid}: affordable_now but earliest_date_for_full_payment "
                    f"{row['earliest_date_for_full_payment']!r} != request_date {req_date!r}"
                )

        changes = row["spending_changes_needed"]
        if changes != "none":
            entries = changes.split("|")
            if len(entries) > 3 or not all(CHANGE_ENTRY_RE.match(e) for e in entries):
                errors.append(f"{rid}: malformed spending_changes_needed {changes!r}")
            stopped = {e.split(":")[1] for e in entries if e.startswith("stop:")}
            reduced = {e.split(":")[1] for e in entries if e.startswith("reduce_to:")}
            if stopped & reduced:
                errors.append(f"{rid}: same event both stopped and reduced: {stopped & reduced}")

        if row["recommended_payment_method"] == "partial_payment":
            entries = plan.split("|")
            if len(entries) != 2:
                errors.append(f"{rid}: partial_payment must have exactly 2 payments, got {plan!r}")
            elif requested is not None:
                total = sum(float(e.split(":")[1]) for e in entries)
                if abs(total - requested) > 1e-2:
                    errors.append(f"{rid}: partial_payment total {total} != requested_amount {requested}")

        if not row["decision_explanation"].strip():
            errors.append(f"{rid}: empty decision_explanation")

    print(f"=== validate_output_schema ({output_path}) ===")
    print(f"rows checked: {len(out)}")
    if errors:
        print(f"{len(errors)} issue(s) found:")
        for e in errors[:30]:
            print(f"  - {e}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more")
    else:
        print("No schema/consistency issues found.")
    print()
    return errors


if __name__ == "__main__":
    score_against_samples()
    validate_output_schema()
