import pm4py
import pandas as pd
import matplotlib.pyplot as plt

log = pm4py.read_xes("Supermarket_data/Supermarket_Clerk.xes")
df = pm4py.convert_to_dataframe(log)

df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)

price_checks = df[df['concept:name'] == 'Start Price Check'].copy()

price_checks = price_checks.dropna(subset=['id'])
price_checks['id'] = price_checks['id'].astype(int)

product_price_check_counts = (
    price_checks
    .groupby('id')
    .size()
    .reset_index(name='Price Check Count')
    .sort_values('Price Check Count', ascending=False)
)

print("--- Price Checks per Product ---")
print(product_price_check_counts)

TOP_N = 20
top_products = product_price_check_counts.head(TOP_N)

plt.figure(figsize=(14, 6))
plt.bar(
    top_products['id'].astype(str),
    top_products['Price Check Count'],
    color='steelblue',
    edgecolor='navy'
)
plt.title(f'Top {TOP_N} Products by Number of Price Checks Required')
plt.xlabel('Product ID')
plt.ylabel('Number of Price Checks')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(
    product_price_check_counts['Price Check Count'],
    bins=20,
    color='coral',
    edgecolor='darkred'
)
plt.title('Distribution of Price Check Counts Across All Products')
plt.xlabel('Number of Price Checks')
plt.ylabel('Number of Products')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()