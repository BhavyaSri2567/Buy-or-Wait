"""Load all dataset/*.csv files with consistent typing."""
from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


def _p(dataset_dir: str, name: str) -> str:
    return os.path.join(dataset_dir, name)


@dataclass
class Dataset:
    requests: pd.DataFrame
    sample_requests: pd.DataFrame
    profiles: pd.DataFrame
    events: pd.DataFrame
    exchange_rates: pd.DataFrame
    payment_options: pd.DataFrame
    messages: pd.DataFrame
    images: pd.DataFrame
    dataset_dir: str


def load_dataset(dataset_dir: str) -> Dataset:
    requests = pd.read_csv(_p(dataset_dir, "requests.csv"))
    for col in ("request_date", "desired_completion_date"):
        requests[col] = pd.to_datetime(requests[col]).dt.date

    sample_requests = pd.read_csv(_p(dataset_dir, "sample_requests.csv"))

    profiles = pd.read_csv(_p(dataset_dir, "financial_profiles.csv"))

    events = pd.read_csv(_p(dataset_dir, "financial_events.csv"))
    for col in ("event_date", "settlement_date"):
        events[col] = pd.to_datetime(events[col], errors="coerce").dt.date

    exchange_rates = pd.read_csv(_p(dataset_dir, "exchange_rates.csv"))

    payment_options = pd.read_csv(_p(dataset_dir, "request_payment_options.csv"))
    payment_options["first_payment_date"] = pd.to_datetime(
        payment_options["first_payment_date"]
    ).dt.date

    messages = pd.read_csv(_p(dataset_dir, "messages.csv"))
    images = pd.read_csv(_p(dataset_dir, "images.csv"))

    return Dataset(
        requests=requests,
        sample_requests=sample_requests,
        profiles=profiles,
        events=events,
        exchange_rates=exchange_rates,
        payment_options=payment_options,
        messages=messages,
        images=images,
        dataset_dir=dataset_dir,
    )
