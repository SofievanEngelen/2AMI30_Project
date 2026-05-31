import pm4py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

log = pm4py.read_xes("Supermarket_data/Supermarket_Customer.xes")
df  = pm4py.convert_to_dataframe(log)
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)

enter_q = (
    df[df['concept:name'] == 'Enter Queue']
    [['case:concept:name', 'time:timestamp', 'mC', 'mI', 'cid']]
    .rename(columns={'time:timestamp': 't_enter',
                     'mC': 'cust_ahead',
                     'mI': 'items_ahead'})
)

abandon = (
    df[df['concept:name'] == 'Abandon cart and leave']
    [['case:concept:name', 'time:timestamp', 'mC', 'mI']]
    .rename(columns={'time:timestamp': 't_abandon',
                     'mC': 'ab_cust',
                     'mI': 'ab_items'})
)


cases = enter_q.merge(abandon, on='case:concept:name', how='left')
cases['abandoned'] = cases['t_abandon'].notna()
cases['wait_s']    = (cases['t_abandon'] - cases['t_enter']).dt.total_seconds()
cases = cases.dropna(subset=['cust_ahead', 'items_ahead'])

n, n_ab  = len(cases), int(cases['abandoned'].sum())
overall  = n_ab / n * 100

print(f"Customers who joined a queue : {n:,}")
print(f"Abandoned                    : {n_ab:,}  ({overall:.1f}%)")
print("\n--- Queue state at entry: abandoners vs completers ---")
print(cases.groupby('abandoned')[['cust_ahead', 'items_ahead']]
      .agg(['mean', 'median', 'max']).round(1))

med_c = cases['cust_ahead'].median()
med_i = cases['items_ahead'].median()

SCEN = {(False, False): 'Low cust / Low items',
        (False, True):  'Low cust / High items',
        (True,  False): 'High cust / Low items',
        (True,  True):  'High cust / High items'}
SCEN_ORDER  = list(SCEN.values())
SCEN_COLORS = ['#4daf4a', '#ff7f00', '#377eb8', '#e41a1c']
SCEN_SHORT  = ['LC/LI', 'LC/HI', 'HC/LI', 'HC/HI']

cases['scenario'] = [
    SCEN[(c > med_c, i > med_i)]
    for c, i in zip(cases['cust_ahead'], cases['items_ahead'])
]

scen_stats = (
    cases.groupby('scenario')['abandoned']
    .agg(n='count', n_ab='sum', rate='mean')
    .assign(rate=lambda x: x['rate'] * 100)
    .reindex(SCEN_ORDER)
)

print(f"\nMedian customers ahead : {med_c:.1f}")
print(f"Median items ahead     : {med_i:.1f}")
print("\n--- Abandonment rate by scenario ---")
print(scen_stats.round(2))

MAX_C = int(cases['cust_ahead'].quantile(0.99))
MAX_I = int(cases['items_ahead'].quantile(0.99))
BIN_I = max(5, MAX_I // 12)

C_EDGES = list(range(0, MAX_C + 2))
I_EDGES = list(range(0, MAX_I + BIN_I + 1, BIN_I))
C_LABELS = [str(e) for e in C_EDGES[:-1]]
I_LABELS = [str(e) for e in I_EDGES[:-1]]

def rate_by_bin(subset, col, edges, labels, min_n=8):
    cut = pd.cut(subset[col], bins=edges, right=False, labels=labels)
    grp = subset.groupby(cut, observed=False)['abandoned'].agg(rate='mean', n='count')
    grp['rate'] = grp['rate'].where(grp['n'] >= min_n)
    return grp

c_rate = rate_by_bin(cases, 'cust_ahead',  C_EDGES, C_LABELS)
i_rate = rate_by_bin(cases, 'items_ahead', I_EDGES, I_LABELS)

fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('Abandonment Threshold Analysis — Queue State at Queue Entry',
              fontsize=15, fontweight='bold')

comp  = cases[~cases['abandoned']]
aband = cases[ cases['abandoned']]

ax = axes[0, 0]
ax.scatter(comp['cust_ahead'],  comp['items_ahead'],
           alpha=0.2, s=8,  c='steelblue', label=f'Completed  n={len(comp):,}')
ax.scatter(aband['cust_ahead'], aband['items_ahead'],
           alpha=0.5, s=14, c='firebrick', label=f'Abandoned  n={len(aband):,}')
ax.set_xlabel('Customers Ahead at Entry')
ax.set_ylabel('Remaining Items Ahead at Entry')
ax.set_title('Queue State at Entry: Completed vs Abandoned')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[0, 1]
vc = c_rate.dropna()
ax.bar(vc.index.astype(float), vc['rate'] * 100,
       width=0.75, color='coral', edgecolor='black', alpha=0.85)
for lbl, row in vc.iterrows():
    ax.text(float(lbl), row['rate'] * 100 + 0.5,
            f"n={int(row['n'])}", ha='center', fontsize=7)
ax.axhline(overall, color='navy', ls='--', lw=1.2, label=f'Overall {overall:.1f}%')
ax.set_xlabel('Customers Ahead at Entry')
ax.set_ylabel('Abandonment Rate (%)')
ax.set_title('Abandonment Rate by Customers Ahead')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.4)

ax = axes[1, 0]
vi   = i_rate.dropna()
xs_i = range(len(vi))
ax.bar(xs_i, vi['rate'] * 100, color='darkorange', edgecolor='black', alpha=0.85)
for x, (lbl, row) in enumerate(vi.iterrows()):
    ax.text(x, row['rate'] * 100 + 0.5, f"n={int(row['n'])}",
            ha='center', fontsize=7, rotation=45)
ax.set_xticks(xs_i); ax.set_xticklabels(vi.index, rotation=45, fontsize=8)
ax.axhline(overall, color='navy', ls='--', lw=1.2, label=f'Overall {overall:.1f}%')
ax.set_xlabel(f'Remaining Items Ahead at Entry (bins of {BIN_I})')
ax.set_ylabel('Abandonment Rate (%)')
ax.set_title('Abandonment Rate by Items Ahead')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.4)

ax = axes[1, 1]
N_BIN = 7
c_q = np.unique(np.percentile(cases['cust_ahead'],  np.linspace(0, 100, N_BIN + 1)))
i_q = np.unique(np.percentile(cases['items_ahead'], np.linspace(0, 100, N_BIN + 1)))
cases['hm_c'] = pd.cut(cases['cust_ahead'],  bins=c_q, include_lowest=True, labels=False)
cases['hm_i'] = pd.cut(cases['items_ahead'], bins=i_q, include_lowest=True, labels=False)
pivot = (
    cases.groupby(['hm_c', 'hm_i'], observed=True)['abandoned']
    .mean()
    .unstack(fill_value=np.nan)
    .reindex(index=range(len(c_q)-1), columns=range(len(i_q)-1))
)
cmap_hm = plt.cm.YlOrRd.copy()
cmap_hm.set_bad(color='lightgrey')   # missing cells shown in grey
im = ax.imshow(np.ma.masked_invalid(pivot.values.T), aspect='auto',
               cmap=cmap_hm, origin='lower', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Abandonment Rate')
ax.set_xticks(range(len(c_q)-1))
ax.set_xticklabels([f'{int(c_q[k])}–{int(c_q[k+1])}' for k in range(len(c_q)-1)],
                   rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(i_q)-1))
ax.set_yticklabels([f'{int(i_q[k])}–{int(i_q[k+1])}' for k in range(len(i_q)-1)],
                   fontsize=7)
ax.set_xlabel('Customers Ahead (range)'); ax.set_ylabel('Items Ahead (range)')
ax.set_title('2D Heatmap: Abandonment Rate\nby (Customers, Items) Ahead at Entry')

plt.tight_layout()
plt.savefig('Supermarket_data/h12_join_time.png', dpi=150, bbox_inches='tight')
plt.show()

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('Queue Scenario Analysis: Low/High Customers × Low/High Items\n'
              f'(split at median: {int(med_c)} customers, {int(med_i)} items)',
              fontsize=13, fontweight='bold')

ax = axes2[0]
scen_stats_filled = scen_stats.fillna(0)
rates_s = [scen_stats_filled.loc[s, 'rate'] for s in SCEN_ORDER]
ns_s    = [scen_stats_filled.loc[s, 'n']    for s in SCEN_ORDER]
bars    = ax.bar(SCEN_SHORT, rates_s, color=SCEN_COLORS, edgecolor='black', alpha=0.85)
for bar, n_val, r in zip(bars, ns_s, rates_s):
    label = f'n={int(n_val):,}' if n_val > 0 else 'n=0'
    ax.text(bar.get_x() + bar.get_width()/2, r + 0.3,
            label, ha='center', fontsize=9)
ax.axhline(overall, color='black', ls='--', lw=1.3, label=f'Overall {overall:.1f}%')
ax.set_ylabel('Abandonment Rate (%)')
ax.set_title('Abandonment Rate per Scenario\n'
             '(LC/HC = Low/High Cust  |  LI/HI = Low/High Items)')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.4)

ax = axes2[1]
for grp_lbl, mask, color, ls in [
    (f'Low cust  (≤{int(med_c)})',  cases['cust_ahead'] <= med_c, 'steelblue', '-'),
    (f'High cust (>{int(med_c)})',  cases['cust_ahead'] >  med_c, 'firebrick', '--'),
]:
    sub   = rate_by_bin(cases[mask].copy(), 'items_ahead', I_EDGES, I_LABELS)
    xs    = np.arange(len(sub))
    valid = sub['rate'].notna()
    ax.plot(xs[valid], sub['rate'].values[valid] * 100,
            marker='o', color=color, ls=ls, lw=2, label=grp_lbl)
ax.set_xticks(range(len(I_LABELS)))
ax.set_xticklabels(I_LABELS, rotation=45, fontsize=8)
ax.axhline(overall, color='grey', ls=':', lw=1, label=f'Overall {overall:.1f}%')
ax.set_xlabel(f'Remaining Items Ahead at Entry (bins of {BIN_I})')
ax.set_ylabel('Abandonment Rate (%)')
ax.set_title('Items Ahead → Abandonment Rate\n(stratified by customer load)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes2[2]
for grp_lbl, mask, color, ls in [
    (f'Low items  (≤{int(med_i)})', cases['items_ahead'] <= med_i, 'steelblue', '-'),
    (f'High items (>{int(med_i)})', cases['items_ahead'] >  med_i, 'firebrick', '--'),
]:
    sub   = rate_by_bin(cases[mask].copy(), 'cust_ahead', C_EDGES, C_LABELS)
    valid = sub['rate'].notna()
    ax.plot(sub.index.values[valid].astype(float),
            sub['rate'].values[valid] * 100,
            marker='o', color=color, ls=ls, lw=2, label=grp_lbl)
ax.axhline(overall, color='grey', ls=':', lw=1, label=f'Overall {overall:.1f}%')
ax.set_xlabel('Customers Ahead at Entry')
ax.set_ylabel('Abandonment Rate (%)')
ax.set_title('Customers Ahead → Abandonment Rate\n(stratified by item load)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Supermarket_data/h12_scenarios.png', dpi=150, bbox_inches='tight')
plt.show()

aband_df = cases[cases['abandoned']].copy()
has_dyn  = (aband_df['ab_cust'].notna().mean()  > 0.1 and
            aband_df['ab_items'].notna().mean() > 0.1)

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle('Queue Dynamics for Abandoning Customers', fontsize=14, fontweight='bold')

ax = axes3[0]
valid_t = aband_df.dropna(subset=['wait_s'])
valid_t = valid_t[valid_t['wait_s'] >= 0]
ax.hist(valid_t['wait_s'], bins=60, color='firebrick', edgecolor='black', alpha=0.8)
mean_w, med_w = valid_t['wait_s'].mean(), valid_t['wait_s'].median()
ax.axvline(mean_w, color='black', ls='--', lw=1.5, label=f'Mean   {mean_w:.0f}s')
ax.axvline(med_w,  color='navy',  ls=':',  lw=1.5, label=f'Median {med_w:.0f}s')
ax.set_xlabel('Time in Queue Before Abandonment (seconds)')
ax.set_ylabel('Number of Customers')
ax.set_title('How Long Did Abandoners Wait Before Giving Up?')
ax.legend(); ax.grid(axis='y', alpha=0.4)

ax = axes3[1]
if has_dyn:
    dyn    = aband_df.dropna(subset=['ab_cust', 'ab_items'])
    sample = dyn.sample(min(400, len(dyn)), random_state=42)
    ax.scatter(dyn['cust_ahead'], dyn['items_ahead'],
               alpha=0.5, s=25, c='steelblue', label='At Entry', zorder=2)
    ax.scatter(dyn['ab_cust'], dyn['ab_items'],
               alpha=0.5, s=25, c='firebrick', marker='X', label='At Abandonment', zorder=3)
    for _, r in sample.iterrows():
        ax.annotate('', xy=(r['ab_cust'], r['ab_items']),
                    xytext=(r['cust_ahead'], r['items_ahead']),
                    arrowprops=dict(arrowstyle='->', color='grey', alpha=0.2, lw=0.5))
    ax.set_xlabel('Customers Ahead'); ax.set_ylabel('Remaining Items Ahead')
    ax.set_title('Queue State: Entry → Abandonment\n(arrows = evolution while waiting)')
    ax.legend(); ax.grid(alpha=0.3)
else:
    valid_w = valid_t.copy()
    q_lbls  = ['Q1 fastest wait', 'Q2', 'Q3', 'Q4 slowest wait']
    valid_w['wq'] = pd.qcut(valid_w['wait_s'], q=4, labels=q_lbls)
    for lbl, col in zip(q_lbls, ['#4daf4a', '#ff7f00', '#377eb8', '#e41a1c']):
        sub = valid_w[valid_w['wq'] == lbl]
        ax.scatter(sub['cust_ahead'], sub['items_ahead'],
                   alpha=0.45, s=15, c=col, label=lbl)
    ax.set_xlabel('Customers Ahead at Entry')
    ax.set_ylabel('Remaining Items Ahead at Entry')
    ax.set_title('Abandoners at Entry State, Coloured by Wait-Time Quartile\n'
                 '(dynamic state not available in log)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Supermarket_data/h12_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()