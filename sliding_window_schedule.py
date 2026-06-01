"""
Sliding-window risk analysis for fixed cashier schedule improvement.

This script starts from the original supermarket XES logs and produces:
- customer_log.parquet or customer_log.csv
- queue_entry_cases.csv
- cashier_schedule.csv
- sliding_windows.csv
- recurring_risk_by_weekday_time.csv
- schedule_reallocation_candidates.csv
- sliding_window_risk_score.png
- sliding_window_risk_labels.png
- recurring_high_risk_heatmap.png
- daily cashier roster plots

It uses caching so the full customer XES log does not have to be parsed every time.

Run with:
    python sliding_window_schedule_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pm4py


# =============================================================================
# HARD-CODED SETTINGS
# =============================================================================

CUSTOMER_LOG_PATH = "Supermarket_data/Supermarket_Customer.xes"
CASHIER_LOG_PATH = "Supermarket_data/Supermarket_Cashier.xes"
OUTPUT_DIR = "Supermarket_data/sliding_window_results"

WINDOW_MINUTES = 60
STEP_MINUTES = 15
MIN_QUEUE_ENTRIES = 30

ITEM_THRESHOLD = 72
CUSTOMER_THRESHOLD = 3

HIGH_RISK_SHARE_THRESHOLD = 0.50
LOW_RISK_SHARE_THRESHOLD = 0.50

# Require repeated observations before calling a time period recurring.
MIN_RECURRING_WINDOWS = 4

# Cache settings
USE_CACHE = True

# Set this to True if you want to reread the original customer XES file.
REBUILD_CUSTOMER_CACHE = False

# Set this to True if you changed build_queue_cases().
REBUILD_CASES_CACHE = False

# Set this to True if you want to reread the original cashier XES file.
REBUILD_CASHIER_CACHE = False

# Parquet is faster and better for the full customer log.
# If you do not have pyarrow installed, set this to False.
USE_PARQUET_FOR_CUSTOMER_CACHE = True


DAY_INDEX = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

DAY_ORDER = list(DAY_INDEX.keys())


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def read_xes(path):
    log = pm4py.read_xes(path)
    df = pm4py.convert_to_dataframe(log)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    return df


def save_customer_log_cache(customer_df, cache_path):
    if cache_path.endswith(".parquet"):
        customer_df.to_parquet(cache_path, index=False)
    else:
        customer_df.to_csv(cache_path, index=False)


def load_customer_log_cache(cache_path):
    if cache_path.endswith(".parquet"):
        customer_df = pd.read_parquet(cache_path)
    else:
        customer_df = pd.read_csv(cache_path)

    customer_df["time:timestamp"] = pd.to_datetime(
        customer_df["time:timestamp"],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    customer_df = customer_df.dropna(subset=["time:timestamp"])

    return customer_df


def load_cases_cache(cases_path):
    cases = pd.read_csv(cases_path)

    cases["t_enter"] = pd.to_datetime(
        cases["t_enter"],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    if "t_abandon" in cases.columns:
        cases["t_abandon"] = pd.to_datetime(
            cases["t_abandon"],
            utc=True,
            format="mixed",
            errors="coerce",
        )

    if "abandoned" in cases.columns:
        cases["abandoned"] = (
            cases["abandoned"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    cases = cases.dropna(subset=["t_enter"])

    return cases


def load_schedule_cache(schedule_path):
    schedule = pd.read_csv(schedule_path)

    if "Shift Start Timestamp" in schedule.columns:
        schedule["Shift Start Timestamp"] = pd.to_datetime(
            schedule["Shift Start Timestamp"],
            utc=True,
            format="mixed",
            errors="coerce",
        )

    if "Shift End Timestamp" in schedule.columns:
        schedule["Shift End Timestamp"] = pd.to_datetime(
            schedule["Shift End Timestamp"],
            utc=True,
            format="mixed",
            errors="coerce",
        )

    schedule = schedule.dropna(
        subset=["Shift Start Timestamp", "Shift End Timestamp"]
    )

    return schedule


# =============================================================================
# 1. BUILD QUEUE-ENTRY CASES
# =============================================================================

def build_queue_cases(customer_df):
    """
    Builds one row per queue-entry observation.

    Uses:
    - mC = customers ahead
    - mI = items ahead
    - cid = counter ID
    """

    enter_q = (
        customer_df[customer_df["concept:name"] == "Enter Queue"]
        [["case:concept:name", "time:timestamp", "mC", "mI", "cid"]]
        .rename(
            columns={
                "time:timestamp": "t_enter",
                "mC": "cust_ahead",
                "mI": "items_ahead",
            }
        )
    )

    abandon = (
        customer_df[customer_df["concept:name"] == "Abandon cart and leave"]
        [["case:concept:name", "time:timestamp", "mC", "mI"]]
        .rename(
            columns={
                "time:timestamp": "t_abandon",
                "mC": "ab_cust",
                "mI": "ab_items",
            }
        )
    )

    cases = enter_q.merge(abandon, on="case:concept:name", how="left")

    cases["abandoned"] = cases["t_abandon"].notna()
    cases["wait_s"] = (cases["t_abandon"] - cases["t_enter"]).dt.total_seconds()

    cases = cases.dropna(subset=["cust_ahead", "items_ahead"]).copy()

    cases["t_enter"] = pd.to_datetime(cases["t_enter"], utc=True)
    cases["cust_ahead"] = pd.to_numeric(cases["cust_ahead"], errors="coerce")
    cases["items_ahead"] = pd.to_numeric(cases["items_ahead"], errors="coerce")

    cases = cases.dropna(subset=["cust_ahead", "items_ahead"]).copy()

    cases["weekday"] = cases["t_enter"].dt.day_name()
    cases["day_index"] = cases["weekday"].map(DAY_INDEX)
    cases["hour"] = cases["t_enter"].dt.hour
    cases["date"] = cases["t_enter"].dt.date.astype(str)
    cases["time_of_day"] = cases["t_enter"].dt.strftime("%H:%M")

    return cases.sort_values("t_enter").reset_index(drop=True)


def get_queue_cases_with_cache():
    """
    Fastest route:
    1. Load queue_entry_cases.csv if it exists.
    2. Otherwise load cached full customer log.
    3. Otherwise parse original XES and save cache.
    """

    if USE_PARQUET_FOR_CUSTOMER_CACHE:
        customer_cache_path = os.path.join(OUTPUT_DIR, "customer_log.parquet")
    else:
        customer_cache_path = os.path.join(OUTPUT_DIR, "customer_log.csv")

    cases_path = os.path.join(OUTPUT_DIR, "queue_entry_cases.csv")

    if USE_CACHE and os.path.exists(cases_path) and not REBUILD_CASES_CACHE:
        print("Loading cached queue-entry cases...")
        cases = load_cases_cache(cases_path)
        return cases

    if (
        USE_CACHE
        and os.path.exists(customer_cache_path)
        and not REBUILD_CUSTOMER_CACHE
    ):
        print("Loading cached converted customer log...")
        customer_df = load_customer_log_cache(customer_cache_path)

    else:
        print("Reading original customer XES log...")
        customer_df = read_xes(CUSTOMER_LOG_PATH)

        if USE_CACHE:
            print("Saving converted customer log cache...")
            save_customer_log_cache(customer_df, customer_cache_path)

    print("Building queue-entry cases...")
    cases = build_queue_cases(customer_df)

    if USE_CACHE:
        print("Saving queue-entry cases cache...")
        cases.to_csv(cases_path, index=False)

    return cases


# =============================================================================
# 2. CASHIER SCHEDULE EXTRACTION
# =============================================================================

def extract_cashier_schedule(cashier_df):
    """
    Extracts cashier shifts from the cashier log.

    Supports both:
    - End shift
    - End Shift
    """

    records = []

    for cashier_id, group in cashier_df.groupby("case:concept:name"):
        group = group.sort_values("time:timestamp")

        starts = group[group["concept:name"] == "Start Shift"].reset_index(drop=True)

        ends = group[
            group["concept:name"].isin(["End shift", "End Shift"])
        ].reset_index(drop=True)

        for i in range(min(len(starts), len(ends))):
            start_ts = starts.iloc[i]["time:timestamp"]
            end_ts = ends.iloc[i]["time:timestamp"]

            if pd.isna(start_ts) or pd.isna(end_ts):
                continue

            if end_ts <= start_ts:
                continue

            day_name = start_ts.strftime("%A")

            records.append(
                {
                    "Cashier ID": cashier_id,
                    "Day Name": day_name,
                    "Day Index": DAY_INDEX[day_name],
                    "Shift Start Timestamp": start_ts,
                    "Shift End Timestamp": end_ts,
                    "Start Time": start_ts.strftime("%H:%M:%S"),
                    "End Time": end_ts.strftime("%H:%M:%S"),
                    "Duration (min)": round(
                        (end_ts - start_ts).total_seconds() / 60, 1
                    ),
                }
            )

    schedule = pd.DataFrame(records)

    if schedule.empty:
        return schedule

    return (
        schedule
        .sort_values(["Shift Start Timestamp", "Cashier ID"])
        .reset_index(drop=True)
    )


def get_cashier_schedule_with_cache():
    schedule_path = os.path.join(OUTPUT_DIR, "cashier_schedule.csv")

    if USE_CACHE and os.path.exists(schedule_path) and not REBUILD_CASHIER_CACHE:
        print()
        print("Loading cached cashier schedule...")
        schedule = load_schedule_cache(schedule_path)
        return schedule

    if os.path.exists(CASHIER_LOG_PATH):
        print()
        print("Reading cashier log...")
        cashier_df = read_xes(CASHIER_LOG_PATH)

        schedule = extract_cashier_schedule(cashier_df)

        if USE_CACHE:
            print("Saving cashier schedule cache...")
            schedule.to_csv(schedule_path, index=False)

        return schedule

    print()
    print("No cashier log found; active cashier counts will be omitted.")
    return pd.DataFrame()


def active_cashiers_at_times(window_ends, schedule):
    """
    Counts how many cashier shifts are active at each sliding-window end time.
    """

    if schedule.empty:
        return [np.nan] * len(window_ends)

    starts = pd.to_datetime(schedule["Shift Start Timestamp"], utc=True)
    ends = pd.to_datetime(schedule["Shift End Timestamp"], utc=True)

    counts = []

    for t in window_ends:
        active_count = int(((starts <= t) & (ends > t)).sum())
        counts.append(active_count)

    return counts


# =============================================================================
# 3. DAILY ROSTER PLOTS
# =============================================================================

def time_to_hours(t_str):
    h, m, s = map(int, str(t_str).split(":"))
    return h + m / 60.0 + s / 3600.0


def plot_daily_rosters(schedule, output_dir):
    """
    Creates one roster plot per weekday.
    """

    if schedule.empty:
        return []

    df = schedule.copy()

    df["Start_Hours"] = df["Start Time"].apply(time_to_hours)
    df["End_Hours"] = df["End Time"].apply(time_to_hours)

    df["Start_Rounded"] = (df["Start_Hours"] * 4).round() / 4
    df["End_Rounded"] = (df["End_Hours"] * 4).round() / 4

    schedule_summary = (
        df.groupby(["Cashier ID", "Day Name", "Day Index"])
        .agg(
            {
                "Start_Rounded": "median",
                "End_Rounded": "median",
            }
        )
        .reset_index()
    )

    generated_files = []

    for day in DAY_ORDER:
        day_data = schedule_summary[schedule_summary["Day Name"] == day]

        if day_data.empty:
            continue

        day_data = day_data.sort_values("Cashier ID", ascending=False)

        plt.figure(figsize=(10, 8))

        y_labels = [
            str(int(cid)) if str(cid).replace(".", "", 1).isdigit() else str(cid)
            for cid in day_data["Cashier ID"]
        ]

        widths = day_data["End_Rounded"] - day_data["Start_Rounded"]
        lefts = day_data["Start_Rounded"]

        plt.barh(y_labels, widths, left=lefts, color="skyblue", edgecolor="navy")

        plt.xlabel("Hour of Day (24h)")
        plt.ylabel("Cashier ID")
        plt.title(f"Daily Schedule: {day}")
        plt.grid(axis="x", linestyle="--", alpha=0.6)

        plt.xlim(8, 22)
        plt.xticks(range(8, 23))

        plt.tight_layout()

        filename = os.path.join(output_dir, f"{day.lower()}_roster.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        generated_files.append(filename)

    return generated_files


# =============================================================================
# 4. SLIDING-WINDOW RISK ANALYSIS
# =============================================================================

def compute_sliding_windows(
    cases,
    schedule=None,
    window_minutes=60,
    step_minutes=15,
    min_queue_entries=30,
    item_threshold=None,
    customer_threshold=3,
):
    """
    Computes risk indicators in sliding windows.

    Example:
    - 60-minute window
    - shifted every 15 minutes

    The risk label is computed window by window, not using fixed hourly aggregation.
    """

    cases = cases.sort_values("t_enter").copy()

    if item_threshold is None:
        item_threshold = cases["items_ahead"].quantile(0.75)

    cases["high_items_ahead"] = cases["items_ahead"] >= item_threshold
    cases["high_customers_ahead"] = cases["cust_ahead"] >= customer_threshold

    start = cases["t_enter"].min().ceil(f"{step_minutes}min")
    end = cases["t_enter"].max().floor(f"{step_minutes}min")

    checkpoints = pd.date_range(
        start=start,
        end=end,
        freq=f"{step_minutes}min",
        tz="UTC",
    )

    window_size = pd.Timedelta(minutes=window_minutes)

    rows = []

    for window_end in checkpoints:
        window_start = window_end - window_size

        sub = cases[
            (cases["t_enter"] > window_start)
            & (cases["t_enter"] <= window_end)
        ]

        if sub.empty:
            continue

        rows.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "date": window_end.date().isoformat(),
                "weekday": window_end.strftime("%A"),
                "day_index": DAY_INDEX[window_end.strftime("%A")],
                "time_of_day": window_end.strftime("%H:%M"),
                "hour": window_end.hour,
                "queue_entries": len(sub),
                "abandonments": int(sub["abandoned"].sum()),
                "abandonment_rate": sub["abandoned"].mean() * 100,
                "avg_customers_ahead": sub["cust_ahead"].mean(),
                "median_customers_ahead": sub["cust_ahead"].median(),
                "avg_items_ahead": sub["items_ahead"].mean(),
                "median_items_ahead": sub["items_ahead"].median(),
                "pct_high_customers_ahead": (
                    sub["high_customers_ahead"].mean() * 100
                ),
                "pct_high_items_ahead": sub["high_items_ahead"].mean() * 100,
                "item_threshold_used": item_threshold,
                "customer_threshold_used": customer_threshold,
            }
        )

    windows = pd.DataFrame(rows)

    if windows.empty:
        raise ValueError(
            "No sliding windows were created. Check timestamps and input data."
        )

    if schedule is not None and not schedule.empty:
        windows["active_cashiers"] = active_cashiers_at_times(
            windows["window_end"], schedule
        )
    else:
        windows["active_cashiers"] = np.nan

    overall_abandonment_rate = cases["abandoned"].mean() * 100
    windows["overall_abandonment_rate"] = overall_abandonment_rate

    reliable = windows[windows["queue_entries"] >= min_queue_entries].copy()

    if reliable.empty:
        item_q75 = windows["avg_items_ahead"].quantile(0.75)
        item_q25 = windows["avg_items_ahead"].quantile(0.25)
        cust_q75 = windows["avg_customers_ahead"].quantile(0.75)
        cust_q25 = windows["avg_customers_ahead"].quantile(0.25)
    else:
        item_q75 = reliable["avg_items_ahead"].quantile(0.75)
        item_q25 = reliable["avg_items_ahead"].quantile(0.25)
        cust_q75 = reliable["avg_customers_ahead"].quantile(0.75)
        cust_q25 = reliable["avg_customers_ahead"].quantile(0.25)

    enough_n = windows["queue_entries"] >= min_queue_entries

    high_mask = (
        enough_n
        & (windows["abandonment_rate"] > overall_abandonment_rate)
        & (
            (windows["avg_items_ahead"] >= item_q75)
            | (windows["avg_customers_ahead"] >= cust_q75)
            | (windows["pct_high_items_ahead"] >= 25)
            | (windows["pct_high_customers_ahead"] >= 10)
        )
    )

    low_mask = (
        enough_n
        & (windows["abandonment_rate"] < overall_abandonment_rate)
        & (windows["avg_items_ahead"] <= item_q25)
        & (windows["avg_customers_ahead"] <= cust_q25)
    )

    windows["risk_label"] = "Medium"
    windows.loc[~enough_n, "risk_label"] = "Unreliable low n"
    windows.loc[high_mask, "risk_label"] = "High"
    windows.loc[low_mask, "risk_label"] = "Low"

    risk_parts = [
        windows["abandonment_rate"].rank(pct=True),
        windows["avg_items_ahead"].rank(pct=True),
        windows["avg_customers_ahead"].rank(pct=True),
        windows["pct_high_items_ahead"].rank(pct=True),
        windows["pct_high_customers_ahead"].rank(pct=True),
    ]

    if windows["active_cashiers"].notna().any():
        # Fewer active cashiers should increase risk.
        risk_parts.append(windows["active_cashiers"].rank(pct=True, ascending=False))

    windows["risk_score"] = sum(risk_parts) / len(risk_parts)

    return windows.sort_values("window_end").reset_index(drop=True)


# =============================================================================
# 5. TRANSLATE SLIDING WINDOWS INTO FIXED-SCHEDULE RECOMMENDATIONS
# =============================================================================

def recurring_schedule_summary(
    windows,
    high_share_threshold=0.50,
    low_share_threshold=0.50,
    min_recurring_windows=4,
):
    """
    Risk labels are computed using sliding windows.

    This function only checks whether High-risk or Low-risk sliding windows
    occur repeatedly at the same weekday and time of day. That makes the result
    usable for a fixed schedule recommendation.
    """

    reliable = windows[windows["risk_label"] != "Unreliable low n"].copy()

    if reliable.empty:
        return pd.DataFrame()

    summary = (
        reliable.groupby(["weekday", "day_index", "time_of_day"])
        .agg(
            windows_observed=("risk_label", "size"),
            high_windows=("risk_label", lambda s: (s == "High").sum()),
            low_windows=("risk_label", lambda s: (s == "Low").sum()),
            medium_windows=("risk_label", lambda s: (s == "Medium").sum()),
            avg_abandonment_rate=("abandonment_rate", "mean"),
            avg_customers_ahead=("avg_customers_ahead", "mean"),
            avg_items_ahead=("avg_items_ahead", "mean"),
            avg_queue_entries=("queue_entries", "mean"),
            avg_active_cashiers=("active_cashiers", "mean"),
            avg_risk_score=("risk_score", "mean"),
        )
        .reset_index()
    )

    summary["high_risk_share"] = (
        summary["high_windows"] / summary["windows_observed"]
    )

    summary["low_risk_share"] = (
        summary["low_windows"] / summary["windows_observed"]
    )

    summary["schedule_label"] = "Keep unchanged / investigate"

    summary.loc[
        (summary["windows_observed"] >= min_recurring_windows)
        & (summary["high_risk_share"] >= high_share_threshold),
        "schedule_label",
    ] = "Target period for extra cashier coverage"

    summary.loc[
        (summary["windows_observed"] >= min_recurring_windows)
        & (summary["low_risk_share"] >= low_share_threshold),
        "schedule_label",
    ] = "Possible source period to move cashier coverage away from"

    return summary.sort_values(["day_index", "time_of_day"]).reset_index(drop=True)


def make_schedule_reallocation_candidates(summary):
    """
    Creates a compact table of the best target and source periods.
    """

    if summary.empty:
        return pd.DataFrame()

    targets = summary[
        summary["schedule_label"] == "Target period for extra cashier coverage"
    ].copy()

    sources = summary[
        summary["schedule_label"]
        == "Possible source period to move cashier coverage away from"
    ].copy()

    targets = targets.sort_values(
        ["high_risk_share", "windows_observed", "avg_risk_score"],
        ascending=[False, False, False],
    ).head(25)

    sources = sources.sort_values(
        ["low_risk_share", "windows_observed", "avg_risk_score"],
        ascending=[False, False, True],
    ).head(25)

    targets["candidate_type"] = "Target period for extra cashier coverage"
    sources["candidate_type"] = (
        "Possible source period to move cashier coverage away from"
    )

    candidates = pd.concat([targets, sources], ignore_index=True)

    cols = [
        "candidate_type",
        "weekday",
        "time_of_day",
        "windows_observed",
        "high_risk_share",
        "low_risk_share",
        "avg_abandonment_rate",
        "avg_customers_ahead",
        "avg_items_ahead",
        "avg_queue_entries",
        "avg_active_cashiers",
        "avg_risk_score",
        "schedule_label",
    ]

    if candidates.empty:
        return candidates

    return candidates[cols]


# =============================================================================
# 6. PLOTS
# =============================================================================

def plot_sliding_risk(windows, output_dir):
    reliable = windows[windows["risk_label"] != "Unreliable low n"].copy()

    if reliable.empty:
        return

    plt.figure(figsize=(14, 5))
    plt.plot(reliable["window_end"], reliable["risk_score"], linewidth=1)
    plt.title("Sliding-window checkout risk score")
    plt.xlabel("Window end time")
    plt.ylabel("Risk score")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "sliding_window_risk_score.png"),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def plot_risk_labels(windows, output_dir):
    reliable = windows[windows["risk_label"] != "Unreliable low n"].copy()

    if reliable.empty:
        return

    y_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    }

    y = reliable["risk_label"].map(y_map)

    plt.figure(figsize=(14, 4.5))
    plt.scatter(reliable["window_end"], y, s=12, alpha=0.7)

    plt.yticks([0, 1, 2], ["Low", "Medium", "High"])
    plt.title("Sliding-window risk labels")
    plt.xlabel("Window end time")
    plt.ylabel("Risk label")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "sliding_window_risk_labels.png"),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def plot_recurring_high_risk_heatmap(summary, output_dir):
    if summary.empty:
        return

    pivot = (
        summary.pivot_table(
            index="weekday",
            columns="time_of_day",
            values="high_risk_share",
            observed=False,
        )
        .reindex(DAY_ORDER)
    )

    plt.figure(figsize=(16, 5))

    plt.imshow(
        pivot.values,
        aspect="auto",
        origin="upper",
        vmin=0,
        vmax=1,
    )

    plt.colorbar(label="Share of repeated windows labelled High risk")

    plt.title("Recurring high-risk share by weekday and time of day")
    plt.xlabel("Time of day")
    plt.ylabel("Weekday")

    n_cols = len(pivot.columns)
    step = max(1, n_cols // 16)
    ticks = np.arange(0, n_cols, step)

    plt.xticks(
        ticks,
        [pivot.columns[i] for i in ticks],
        rotation=45,
        ha="right",
    )

    plt.yticks(np.arange(len(pivot.index)), pivot.index)

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "recurring_high_risk_heatmap.png"),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# 7. MAIN SCRIPT
# =============================================================================

def main():
    ensure_output_dir(OUTPUT_DIR)

    cases = get_queue_cases_with_cache()

    cases_path = os.path.join(OUTPUT_DIR, "queue_entry_cases.csv")
    cases.to_csv(cases_path, index=False)

    n = len(cases)
    n_ab = int(cases["abandoned"].sum())
    overall = n_ab / n * 100

    print()
    print(f"Customers who joined a queue : {n:,}")
    print(f"Abandoned                    : {n_ab:,} ({overall:.1f}%)")

    print()
    print("Queue state at entry: abandoners vs completers")
    print(
        cases.groupby("abandoned")[["cust_ahead", "items_ahead"]]
        .agg(["mean", "median", "max"])
        .round(1)
    )

    schedule = get_cashier_schedule_with_cache()

    if not schedule.empty:
        generated_rosters = plot_daily_rosters(schedule, OUTPUT_DIR)
        print(f"Cashier shifts available     : {len(schedule):,}")
        print(f"Roster plots generated       : {len(generated_rosters):,}")

    print()
    print("Computing sliding windows...")

    windows = compute_sliding_windows(
        cases=cases,
        schedule=schedule,
        window_minutes=WINDOW_MINUTES,
        step_minutes=STEP_MINUTES,
        min_queue_entries=MIN_QUEUE_ENTRIES,
        item_threshold=ITEM_THRESHOLD,
        customer_threshold=CUSTOMER_THRESHOLD,
    )

    windows_path = os.path.join(OUTPUT_DIR, "sliding_windows.csv")
    windows.to_csv(windows_path, index=False)

    print("Creating recurring risk summary...")

    summary = recurring_schedule_summary(
        windows,
        high_share_threshold=HIGH_RISK_SHARE_THRESHOLD,
        low_share_threshold=LOW_RISK_SHARE_THRESHOLD,
        min_recurring_windows=MIN_RECURRING_WINDOWS,
    )

    summary_path = os.path.join(OUTPUT_DIR, "recurring_risk_by_weekday_time.csv")
    summary.to_csv(summary_path, index=False)

    candidates = make_schedule_reallocation_candidates(summary)

    candidates_path = os.path.join(OUTPUT_DIR, "schedule_reallocation_candidates.csv")
    candidates.to_csv(candidates_path, index=False)

    print("Creating plots...")

    plot_sliding_risk(windows, OUTPUT_DIR)
    plot_risk_labels(windows, OUTPUT_DIR)
    plot_recurring_high_risk_heatmap(summary, OUTPUT_DIR)

    print()
    print("Saved outputs to:")
    print(os.path.abspath(OUTPUT_DIR))

    print()
    print("Important files:")
    print("- customer_log.parquet or customer_log.csv")
    print("- queue_entry_cases.csv")
    print("- cashier_schedule.csv")
    print("- sliding_windows.csv")
    print("- recurring_risk_by_weekday_time.csv")
    print("- schedule_reallocation_candidates.csv")
    print("- sliding_window_risk_score.png")
    print("- sliding_window_risk_labels.png")
    print("- recurring_high_risk_heatmap.png")
    print("- monday_roster.png, tuesday_roster.png, etc.")

    if not candidates.empty:
        print()
        print("Top schedule reallocation candidates:")
        print(candidates.head(15).to_string(index=False))
    else:
        print()
        print("No schedule candidates found with the current thresholds.")


if __name__ == "__main__":
    main()