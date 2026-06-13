import ast
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pm4py
import re

CUSTOMER_LOG = sys.argv[1]
STATE_LOGS = sys.argv[2:] if len(sys.argv) > 2 else [CUSTOMER_LOG]

OUT_DIR        = "Supermarket_data"
RATE_PLOT      = os.path.join(OUT_DIR, "abandon_rate_by_people_at_arrival.png")
ABANDON_HIST   = os.path.join(OUT_DIR, "queue_state_at_abandon.png")

ENTER_STORE = "Enter store"
ABANDON     = "Abandon cart and leave"
ARRIVE      = "Go to Checkout"
ENTER_QUEUE = "Enter Queue"
CASE        = "case:concept:name"

def parse_col(v):
    if not isinstance(v, str):
        return None

    matches = re.findall(
        r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(True|False|true|false)\s*\)',
        v
    )

    return {
        int(cid): (
            int(n),
            int(it),
            op.lower() == "true"
        )
        for cid, n, it, op in matches
    }

def open_queue_stats(col):
    if col is None:
        return np.nan, np.nan
    open_lens = [n for (n, _it, is_open) in col.values() if is_open]
    if not open_lens:
        return np.nan, np.nan
    return float(np.mean(open_lens)), float(np.max(open_lens))

def load(path):
    df = pm4py.convert_to_dataframe(pm4py.read_xes(path))
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    return df

parts = []
for p in STATE_LOGS:
    d = load(p)
    if "col" not in d.columns:
        continue
    d = d[["time:timestamp", "col"]].copy()
    d["col_dict"] = d["col"].apply(parse_col)
    d = d[d["col_dict"].notna()].copy()
    d["src"] = os.path.basename(p)
    parts.append(d[["time:timestamp", "col_dict", "src"]])

if not parts:
    sys.exit("No `col` snapshots found in the state logs -- cannot read queue state.")

snap = pd.concat(parts).sort_values("time:timestamp").reset_index(drop=True)
snap_ts = snap["time:timestamp"].to_numpy("datetime64[ns]").astype("int64")

def lookup(ts_series):
    keys = ts_series.to_numpy("datetime64[ns]").astype("int64")
    pos = np.searchsorted(snap_ts, keys, side="right") - 1
    q_mean, q_max, stale, src = [], [], [], []
    for k, ip in zip(keys, pos):
        if ip < 0:
            q_mean.append(np.nan); q_max.append(np.nan)
            stale.append(np.nan); src.append(None)
        else:
            m, mx = open_queue_stats(snap["col_dict"].iloc[ip])
            q_mean.append(m); q_max.append(mx)
            stale.append((k - snap_ts[ip]) / 1e9)
            src.append(snap["src"].iloc[ip])
    return pd.DataFrame(
        {"q_mean": q_mean, "q_max": q_max, "staleness_s": stale, "src": src},
        index=ts_series.index,
    )

cust_df = load(CUSTOMER_LOG).sort_values("time:timestamp").reset_index(drop=True)
n_entered = cust_df[CASE].nunique()

abandoned_cases = set(cust_df.loc[cust_df["concept:name"] == ABANDON, CASE])

def first_event_per_case(df, activity):
    sub = df[df["concept:name"] == activity].sort_values("time:timestamp")
    return sub.groupby(CASE, as_index=False).first()[[CASE, "time:timestamp"]]

arrive = first_event_per_case(cust_df, ARRIVE)
queue  = first_event_per_case(cust_df, ENTER_QUEUE)

decision = arrive.copy()
missing = set(cust_df[CASE].unique()) - set(decision[CASE])
fallback = queue[queue[CASE].isin(missing)]
decision = pd.concat([decision, fallback], ignore_index=True)

decision["abandoned"] = decision[CASE].isin(abandoned_cases)
decision = pd.concat(
    [decision.reset_index(drop=True),
     lookup(decision["time:timestamp"].reset_index(drop=True))],
    axis=1,
)

ab = decision[decision["abandoned"]]

n_no_decision = n_entered - decision[CASE].nunique()

def rate_by_bin(frame, qcol):
    d = frame.dropna(subset=[qcol]).copy()
    d["k"] = d[qcol].round().astype(int)
    g = d.groupby("k")["abandoned"].agg(n_customers="size", n_abandons="sum")
    g["rate"] = g["n_abandons"] / g["n_customers"]
    return g.sort_index()

rate_mean = rate_by_bin(decision, "q_mean")
rate_max  = rate_by_bin(decision, "q_max")

valid = decision.dropna(subset=["q_mean"])
pooled = valid["abandoned"].mean() * 100

ab_events = cust_df[cust_df["concept:name"] == ABANDON].copy().reset_index(drop=True)
ab_events = pd.concat([ab_events, lookup(ab_events["time:timestamp"])], axis=1)
ab_hist = (ab_events.dropna(subset=["q_mean"])
           .assign(k=lambda d: d["q_mean"].round().astype(int))
           .groupby("k").size())

os.makedirs(OUT_DIR, exist_ok=True)

rate_plot = rate_mean[rate_mean["n_abandons"] > 0]
if not rate_plot.empty:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(rate_plot.index, rate_plot["rate"] * 100,
           color="coral", edgecolor="black", width=0.85)
    for k, r in rate_plot.iterrows():
        ax.text(k, r["rate"] * 100, f"{int(r['n_abandons'])}/{int(r['n_customers'])}",
                ha="center", va="bottom", fontsize=8, color="dimgray")
    ax.axhline(pooled, color="navy", ls="--", lw=1.5,
               label=f"Baseline = {pooled:.1f}%")
    ax.legend()
    ax.set_xlabel("Average open queue at arrival at checkout")
    ax.set_ylabel("Abandonment rate (%)")
    ax.set_title("Abandonment Rate vs. Queue State at Arrival")
    ax.set_xticks(rate_mean.index)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RATE_PLOT, dpi=150, bbox_inches="tight")