import pm4py
import pandas as pd
import matplotlib.pyplot as plt

log = pm4py.read_xes("Supermarket_data/Supermarket_Customer.xes")
df = pm4py.convert_to_dataframe(log)
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)

start_pay = (
    df[df['concept:name'] == 'Start Payment']
    [['case:concept:name', 'time:timestamp', 'p']]
    .rename(columns={'time:timestamp': 'start_time'})
)

end_pay = (
    df[df['concept:name'] == 'Complete Payment']
    [['case:concept:name', 'time:timestamp']]
    .rename(columns={'time:timestamp': 'end_time'})
)

start_pay = (
    df[df['concept:name'] == 'Start Payment']
    [['case:concept:name', 'time:timestamp', 'p']]
    .rename(columns={'time:timestamp': 'start_time'})
    .sort_values('start_time')
)

end_pay = (
    df[df['concept:name'] == 'Complete Payment']
    [['case:concept:name', 'time:timestamp']]
    .rename(columns={'time:timestamp': 'end_time'})
    .sort_values('end_time')
)

payments = pd.merge_asof(
    start_pay,
    end_pay,
    left_on='start_time',
    right_on='end_time',
    by='case:concept:name',
    direction='forward',
    tolerance=pd.Timedelta('10min')
)

payments = payments.dropna(subset=['end_time'])

payments['duration_seconds'] = (
    payments['end_time'] - payments['start_time']
).dt.total_seconds()

payments = payments[payments['duration_seconds'] > 0].dropna(subset=['p'])

lower = payments['duration_seconds'].quantile(0.01)
upper = payments['duration_seconds'].quantile(0.99)
payments = payments[payments['duration_seconds'].between(lower, upper)]

payment_methods = sorted(payments['p'].unique())
colors = plt.cm.Set2.colors

fig, axes = plt.subplots(
    1, len(payment_methods),
    figsize=(6 * len(payment_methods), 5),
    sharey=True
)

if len(payment_methods) == 1:
    axes = [axes]

for ax, method, color in zip(axes, payment_methods, colors):
    subset = payments[payments['p'] == method]['duration_seconds']
    ax.hist(subset, bins=300, color=color, edgecolor='black', alpha=0.85)
    ax.set_title(f'Payment Method: {method}', fontsize=13)
    ax.set_xlabel('Duration (seconds)')
    ax.set_ylabel('Number of Customers')
    ax.axvline(subset.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {subset.mean():.1f}s')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.suptitle('Distribution of Payment Duration by Payment Method',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('payment_type_duration_distribution.png', dpi=300, bbox_inches='tight')
plt.show()