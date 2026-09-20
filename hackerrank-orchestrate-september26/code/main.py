#!/usr/bin/env python3
"""
Buy or Wait? — entry point.

Usage:
    python3 code/main.py [--dataset-dir dataset] [--out output.csv]

Reads dataset/*.csv, reconstructs each user's financial state, runs the
90-day safety check, and writes one prediction row per request in
dataset/requests.csv to the repository-root output.csv (default).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from engine.io_utils import load_dataset
from engine.fx import FxTable
from engine.llm import LLMClient
from engine.events import build_profile, UserFinancialModel
from engine.decision import decide

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


def run(dataset_dir: str, out_path: str, usage_report_path: str) -> None:
    t0 = time.time()
    ds = load_dataset(dataset_dir)
    fx = FxTable(ds.exchange_rates)
    llm = LLMClient()

    profiles_by_user = {row["user_id"]: build_profile(row) for _, row in ds.profiles.iterrows()}
    model_cache = {}

    def get_model(user_id: str) -> UserFinancialModel:
        if user_id not in model_cache:
            model_cache[user_id] = UserFinancialModel(
                profile=profiles_by_user[user_id],
                events_df=ds.events,
                images_df=ds.images,
                messages_df=ds.messages,
                fx=fx,
                llm=llm,
                dataset_dir=dataset_dir,
            )
        return model_cache[user_id]

    rows = []
    notes_log = []
    for _, req in ds.requests.iterrows():
        user_id = req["user_id"]
        if user_id not in profiles_by_user:
            rows.append(
                {
                    "request_id": req["request_id"],
                    "amount_safe_to_pay": 0,
                    "affordability_status": "not_affordable",
                    "recommended_payment_method": "not_recommended",
                    "payment_plan": "none",
                    "earliest_date_for_full_payment": "",
                    "spending_changes_needed": "none",
                    "decision_explanation": "No financial profile found for this user.",
                }
            )
            continue
        model = get_model(user_id)
        row = decide(req, model, ds.payment_options)
        rows.append(row)
    for m in model_cache.values():
        notes_log.extend(m.notes)

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    os.makedirs(os.path.dirname(usage_report_path), exist_ok=True)
    with open(usage_report_path, "w") as f:
        f.write("# Token Usage and Cost Report\n\n")
        f.write(f"Run produced: `{out_path}`\n\n")
        f.write(f"Requests processed: {len(rows)}\n\n")
        f.write(f"Wall-clock runtime: {elapsed:.1f}s\n\n")
        f.write("## Model Usage\n\n")
        for line in llm.usage_report_lines():
            f.write(line + "\n")
        f.write("\n## Notes\n\n")
        f.write(
            "The deterministic engine (currency conversion, recurrence detection, "
            "90-day balance simulation, plan ranking) makes no model calls. The "
            "optional Anthropic API is used only to (a) read amounts off images "
            "linked to blank-amount financial events, and (b) interpret any "
            "message that does not match a known template. With "
            "ANTHROPIC_API_KEY unset, both fall back to deterministic heuristics "
            "documented in engine/events.py and engine/extraction.py, and this "
            "run made zero model calls.\n"
        )
        if notes_log:
            f.write("\n## Data-Resolution Notes (sample)\n\n")
            for n in notes_log[:50]:
                f.write(f"- {n}\n")

    print(f"Wrote {len(rows)} predictions to {out_path}")
    print(f"Usage report: {usage_report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--dataset-dir", default=os.path.join(here, "dataset"))
    parser.add_argument("--out", default=os.path.join(here, "output.csv"))
    parser.add_argument(
        "--usage-report",
        default=os.path.join(here, "evaluation", "usage_report.md"),
    )
    args = parser.parse_args()
    run(args.dataset_dir, args.out, args.usage_report)
