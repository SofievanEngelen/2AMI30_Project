import pm4py
import pandas as pd
import matplotlib.pyplot as plt

log = pm4py.read_xes("Supermarket_data/Supermarket_Customer.xes")
df = pm4py.convert_to_dataframe(log)
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)

def analyze_customer(group):
    return pd.Series({
        's': group['s'].dropna().iloc[0] if group['s'].dropna().any() else None,
        'abandoned': 'Abandon cart and leave' in group['concept:name'].values
    })

case_summary = df.groupby('case:concept:name').apply(analyze_customer).dropna(subset=['s'])

stats = case_summary.groupby('s')['abandoned'].agg(['sum', 'count'])
stats.columns = ['abandoned', 'total']
stats['rate'] = stats['abandoned'] / stats['total'] * 100
print(stats.round(2))

fig, ax = plt.subplots(figsize=(8, 5))
colors = [plt.cm.Set2.colors[i] for i in range(len(stats))]
bars = ax.bar(stats.index, stats['rate'], color=colors, edgecolor='black', width=0.5)

for bar, (_, row) in zip(bars, stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{row['rate']:.1f}%\n(n={int(row['total'])})",
            ha='center', va='bottom', fontsize=10)

ax.set_title('Abandonment Rate by Customer Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Customer Type (s)', fontsize=12)
ax.set_ylabel('Abandonment Rate (%)', fontsize=12)
ax.set_ylim(0, stats['rate'].max() * 1.25)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Supermarket_data/abandonment_rate_by_type.png', dpi=150, bbox_inches='tight')
plt.show()