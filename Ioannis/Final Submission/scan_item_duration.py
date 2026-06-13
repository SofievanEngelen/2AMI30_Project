import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pm4py

warnings.filterwarnings("ignore")

COUNTER_LOG = sys.argv[1] if len(sys.argv) > 1 else "Supermarket_data/Supermarket_Counter.xes"

OUT_DIR  = "Supermarket_data"
PLOT     = os.path.join(OUT_DIR, "scan_time_by_cashier.png")

SCAN        = "Scan Item"
OPEN_COUNTER = "Open counter"
CASE        = "case:concept:name"
CUST        = "id"
CAID        = "caid"

MAX_SCAN_SECONDS = 60.0


def load(path):
    df = pm4py.convert_to_dataframe(pm4py.read_xes(path))
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    return df


def per_item_scan_times(df):
    needed = {"concept:name", "time:timestamp", CASE}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"Log is missing required columns: {sorted(missing)}")
    if CAID not in df.columns:
        sys.exit("No `caid` column found -- this is not the counter log.")
    if not (df["concept:name"] == SCAN).any():
        sys.exit("No `Scan Item` events found -- this is not the counter log.")

    rows = []
    for counter_id, trace in df.groupby(CASE, sort=False):
        trace = trace.sort_values("time:timestamp")

        caid = trace[CAID].where(trace["concept:name"] == OPEN_COUNTER).ffill()

        act      = trace["concept:name"].to_numpy()
        ts       = trace["time:timestamp"].to_numpy()
        cust     = (trace[CUST].to_numpy() if CUST in trace.columns
                    else np.full(len(trace), np.nan))
        caid_arr = caid.to_numpy()

        for k in range(1, len(trace)):
            if act[k] != SCAN or act[k - 1] != SCAN:
                continue
            if not (pd.notna(cust[k]) and cust[k] == cust[k - 1]):
                continue
            dur = (ts[k] - ts[k - 1]) / np.timedelta64(1, "s")
            if not (0 < dur <= MAX_SCAN_SECONDS):
                continue
            rows.append((counter_id, caid_arr[k], cust[k], dur))

    out = pd.DataFrame(rows, columns=["counter", CAID, CUST, "scan_s"])
    return out


def main():
    df = load(COUNTER_LOG)
    scans = per_item_scan_times(df)

    scans = scans.dropna(subset=[CAID]).copy()
    scans[CAID] = scans[CAID].astype(int)

    if scans.empty:
        sys.exit("No usable scan intervals found.")

    g = (scans.groupby(CAID)["scan_s"]
         .agg(n_scans="size", mean_s="mean", median_s="median", std_s="std")
         .sort_values("mean_s"))
    g[["mean_s", "median_s", "std_s"]] = g[["mean_s", "median_s", "std_s"]].round(3)

    overall = scans["scan_s"].mean()

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(g) + 3), 6))
    x = np.arange(len(g))
    ax.bar(x, g["mean_s"], yerr=g["std_s"], capsize=4,
           color="steelblue", edgecolor="black", width=0.7)
    ax.axhline(overall, color="firebrick", ls="--", lw=1.5,
               label=f"Overall mean = {overall:.2f} s")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cashier {c}" for c in g.index], rotation=45, ha="right")
    ax.set_ylabel("Average time per scanned item (s)")
    ax.set_title("Average Scan Item Time per Cashier")
    ax.grid(axis="y", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()