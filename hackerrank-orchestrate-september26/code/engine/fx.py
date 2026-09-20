"""
Currency conversion using dataset/exchange_rates.csv.

The dataset only supplies a handful of direct pairs (EUR<->USD, EUR<->ZAR,
USD<->IDR, USD<->INR) as monthly snapshots. To convert between any two of the
five supported currencies (INR, ZAR, IDR, USD, EUR) we build a small graph per
rate-date and multiply rates along the shortest path (e.g. ZAR -> EUR -> USD
-> IDR).

For a target conversion date that has no exact snapshot, we use the closest
available snapshot date that is not later than the target date. If none
exists before it, we fall back to the earliest available snapshot after it.
This keeps behavior deterministic and never invents a rate.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import date
from functools import lru_cache
from typing import Dict, Tuple

import pandas as pd


class FxTable:
    def __init__(self, exchange_rates_df: pd.DataFrame):
        self.raw = exchange_rates_df.copy()
        self.raw["rate_date"] = pd.to_datetime(self.raw["rate_date"]).dt.date
        self.dates = sorted(self.raw["rate_date"].unique())
        # graph[date][(from,to)] = rate
        self.graph: Dict[date, Dict[Tuple[str, str], float]] = defaultdict(dict)
        for _, row in self.raw.iterrows():
            d = row["rate_date"]
            f, t, r = row["from_currency"], row["to_currency"], float(row["rate"])
            self.graph[d][(f, t)] = r
            # keep an implied inverse edge if not explicitly supplied
            self.graph[d].setdefault((t, f), 1.0 / r)

    def _closest_date(self, target: date) -> date:
        if not self.dates:
            raise ValueError("No exchange rate data available")
        earlier = [d for d in self.dates if d <= target]
        if earlier:
            return max(earlier)
        return min(self.dates)

    @lru_cache(maxsize=None)
    def _rate_path(self, rate_date: date, frm: str, to: str) -> float:
        if frm == to:
            return 1.0
        edges = self.graph.get(rate_date, {})
        # BFS over currency graph, multiplying rates along the path
        adjacency: Dict[str, list] = defaultdict(list)
        for (a, b), r in edges.items():
            adjacency[a].append((b, r))
        visited = {frm}
        queue = deque([(frm, 1.0)])
        while queue:
            node, acc = queue.popleft()
            if node == to:
                return acc
            for nxt, r in adjacency.get(node, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, acc * r))
        raise ValueError(f"No FX path from {frm} to {to} on/near {rate_date}")

    def convert(self, amount: float, frm: str, to: str, on: date) -> float:
        if frm == to:
            return amount
        d = self._closest_date(on)
        rate = self._rate_path(d, frm, to)
        return amount * rate
