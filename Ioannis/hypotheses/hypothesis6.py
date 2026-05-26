import pm4py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

log = pm4py.read_xes("Supermarket_data/Supermarket_Cashier.xes")
df = pm4py.convert_to_dataframe(log)
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)

def extract_shift_durations(group):
    starts = group[group['concept:name'] == 'Start Shift'].sort_values('time:timestamp')
    ends   = group[group['concept:name'] == 'End shift'].sort_values('time:timestamp')
    intervals = []
    for i in range(min(len(starts), len(ends))):
        intervals.append({
            'start': starts.iloc[i]['time:timestamp'],
            'end':   ends.iloc[i]['time:timestamp']
        })
    return intervals

all_shifts = []
for name, group in df.groupby('case:concept:name'):
    all_shifts.extend(extract_shift_durations(group))

unique_days = df['time:timestamp'].dt.date.nunique()
shifts_df   = pd.DataFrame(all_shifts)
shifts_df['date']       = shifts_df['start'].dt.date
shifts_df['weekday']    = shifts_df['start'].dt.dayofweek
shifts_df['duration_h'] = (shifts_df['end'] - shifts_df['start']).dt.total_seconds() / 3600

timeline_min = np.zeros(24 * 60)
for _, s in shifts_df.iterrows():
    m_s = s['start'].hour * 60 + s['start'].minute
    m_e = s['end'].hour   * 60 + s['end'].minute
    if m_s < m_e:
        timeline_min[m_s:m_e] += 1
    else:
        timeline_min[m_s:] += 1
        timeline_min[:m_e] += 1

hourly_avg = (timeline_min / unique_days).reshape(24, 60).mean(axis=1)

daily_counts         = shifts_df.groupby('date').size().reset_index()
daily_counts['weekday'] = pd.to_datetime(
    daily_counts['date'].astype(str)
).dt.dayofweek
daily_by_dow = daily_counts.groupby('weekday')[0].mean()
day_labels   = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

shifts_df['iso_week']  = shifts_df['start'].dt.isocalendar().week.astype(int)
shifts_df['year']      = shifts_df['start'].dt.isocalendar().year.astype(int)
shifts_df['year_week'] = (shifts_df['year'].astype(str) + '-W' +
                          shifts_df['iso_week'].astype(str).str.zfill(2))

shifts_per_day = shifts_df.groupby(['year_week', 'date']).size().reset_index(name='n_cashiers')
weekly_avg     = shifts_per_day.groupby('year_week')['n_cashiers'].mean().sort_index()

TEAL  = '#008080'
CORAL = '#E07B5A'
SLATE = '#5A7EA0'

fig, axes = plt.subplots(
    3, 1,
    figsize=(16, 18),
    gridspec_kw={'hspace': 0.55}
)
fig.suptitle('Cashier Availability Profiles', fontsize=17, fontweight='bold', y=0.995)

ax = axes[0]
hours = np.arange(24)
ax.bar(hours, hourly_avg, color=TEAL, alpha=0.75, width=0.8)
ax.plot(hours, hourly_avg, color=TEAL, linewidth=1.8, marker='o', markersize=4, label='Avg. cashiers')
ax.fill_between(hours, hourly_avg, alpha=0.15, color=TEAL)
ax.set_title('Hourly Availability (avg. across all days)', fontsize=13, pad=10)
ax.set_xlabel('Hour of Day (24 h)', labelpad=8)
ax.set_ylabel('Avg. Cashiers Active', labelpad=8)
ax.set_xticks(hours)
ax.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=40, ha='right', fontsize=8.5)
ax.tick_params(axis='x', pad=4)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.margins(x=0.01)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper left')

ax = axes[1]
x = daily_by_dow.index.tolist()
y = daily_by_dow.values
bars = ax.bar(x, y, color=CORAL, alpha=0.75, width=0.55)
ax.plot(x, y, color=CORAL, linewidth=1.8, marker='o', markersize=5)
y_offset = max(y) * 0.015
for bar, val in zip(bars, y):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + y_offset,
        f'{val:.1f}', ha='center', va='bottom', fontsize=9
    )
ax.set_title('Daily Availability (avg. active cashiers per weekday)', fontsize=13, pad=10)
ax.set_xlabel('Day of Week', labelpad=8)
ax.set_ylabel('Avg. Cashiers Active', labelpad=8)
ax.set_xticks(range(len(day_labels)))
ax.set_xticklabels(day_labels, fontsize=10)
ax.set_ylim(0, max(y) * 1.18)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(axis='y', linestyle='--', alpha=0.5)

ax = axes[2]
wx = range(len(weekly_avg))
ax.bar(wx, weekly_avg.values, color=SLATE, alpha=0.75, width=0.7)
ax.plot(wx, weekly_avg.values, color=SLATE, linewidth=1.8, marker='o', markersize=4, label='Avg. cashiers')
ax.fill_between(wx, weekly_avg.values, alpha=0.15, color=SLATE)
ax.set_title('Weekly Availability (avg. active cashiers per day per week)', fontsize=13, pad=10)
ax.set_xlabel('ISO Week', labelpad=8)
ax.set_ylabel('Avg. Cashiers Active', labelpad=8)
ax.set_xticks(list(wx))
ax.set_xticklabels(weekly_avg.index.tolist(), rotation=40, ha='right', fontsize=8.5)
ax.tick_params(axis='x', pad=4)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.margins(x=0.01)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper left')

plt.savefig('cashier_availability_profiles.png', dpi=150, bbox_inches='tight')
plt.show()