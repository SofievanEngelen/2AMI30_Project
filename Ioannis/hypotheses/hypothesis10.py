import pm4py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

log = pm4py.read_xes("Supermarket_data/Supermarket_Clerk.xes")
df = pm4py.convert_to_dataframe(log)
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)

CLERK_ID = 'case:concept:name'
df = df.sort_values([CLERK_ID, 'time:timestamp'])

JOIN_KEYS = ['id', 'cid', CLERK_ID]

starts = (
    df[df['concept:name'] == 'Start Price Check']
    [JOIN_KEYS + ['time:timestamp']]
    .rename(columns={'time:timestamp': 'start_time'})
)
ends = (
    df[df['concept:name'] == 'End price check']
    [JOIN_KEYS + ['time:timestamp']]
    .rename(columns={'time:timestamp': 'end_time'})
)

price_checks = pd.merge(starts, ends, on=JOIN_KEYS)
price_checks = price_checks[price_checks['end_time'] > price_checks['start_time']].copy()
price_checks['duration_s'] = (price_checks['end_time'] - price_checks['start_time']).dt.total_seconds()

P99 = price_checks['duration_s'].quantile(0.99)
price_checks = price_checks[price_checks['duration_s'] <= P99]

cleanups_df = df[df['concept:name'] == 'Cleanup abandoned item'][[CLERK_ID, 'time:timestamp']]
WINDOW = pd.Timedelta('10min')

cleanup_map = {clerk: group['time:timestamp'].values for clerk, group in cleanups_df.groupby(CLERK_ID)}

def count_personal_cleanups(row):
    clerk = row[CLERK_ID]
    t_start = row['start_time']
    
    if clerk not in cleanup_map:
        return 0
    
    personal_ts = cleanup_map[clerk]
    t_min = (t_start - WINDOW).to_datetime64()
    t_max = t_start.to_datetime64()
    
    return np.sum((personal_ts >= t_min) & (personal_ts < t_max))

print("Calculating workload for each individual clerk...")
price_checks['personal_cleanups'] = price_checks.apply(count_personal_cleanups, axis=1)

bins   = [-1, 0, 1, 2, 4, np.inf]
labels = ['0', '1', '2', '3–4', '5+']
price_checks['cleanup_bucket'] = pd.cut(price_checks['personal_cleanups'], bins=bins, labels=labels)

bucket_stats = (
    price_checks.groupby('cleanup_bucket', observed=True)['duration_s']
    .agg(count='count', mean='mean', median='median', std='std')
    .round(2)
)

r, p = stats.pearsonr(price_checks['personal_cleanups'], price_checks['duration_s'])

print("\n--- INDIVIDUAL Clerk Workload Results ---")
print(bucket_stats)
print(f"\nPearson r = {r:.3f},  p-value = {p:.4f}")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Hypothesis: Do an Individual Clerk's Cleanups Delay Their Price Checks?", fontsize=14, fontweight='bold')

axes[0].scatter(price_checks['personal_cleanups'], price_checks['duration_s'], alpha=0.3, color='forestgreen', s=20)
m, b = np.polyfit(price_checks['personal_cleanups'], price_checks['duration_s'], 1)
xs = np.linspace(price_checks['personal_cleanups'].min(), price_checks['personal_cleanups'].max(), 100)
axes[0].plot(xs, m*xs + b, color='red', label=f'r={r:.3f}')
axes[0].set_xlabel('Cleanups by this Clerk (prior 10m)')
axes[0].set_ylabel('Duration (s)')
axes[0].legend()

groups = [price_checks.loc[price_checks['cleanup_bucket'] == lbl, 'duration_s'].dropna() for lbl in labels]
axes[1].boxplot(groups, labels=labels, patch_artist=True, boxprops=dict(facecolor='mediumseagreen', alpha=0.6))
axes[1].set_xlabel('Cleanup Count')
axes[1].set_title('Distribution per Workload')

x_pos = range(len(bucket_stats))
axes[2].bar(x_pos, bucket_stats['mean'], color='mediumseagreen', alpha=0.8, tick_label=labels)
axes[2].errorbar(x=x_pos, y=bucket_stats['mean'], yerr=bucket_stats['std'], fmt='none', color='black', capsize=4)
for i, v in enumerate(bucket_stats['count']):
    axes[2].text(i, bucket_stats['mean'].iloc[i] + 2, f"n={int(v)}", ha='center', fontsize=9)
axes[2].set_title('Mean Duration ± Std Dev')

plt.tight_layout()
plt.show()