import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('cashier_schedule.csv')

def time_to_hours(t_str):
    h, m, s = map(int, t_str.split(':'))
    return h + m/60.0 + s/3600.0

df['Start_Hours'] = df['Start Time'].apply(time_to_hours)
df['End_Hours'] = df['End Time'].apply(time_to_hours)

df['Start_Rounded'] = (df['Start_Hours'] * 4).round() / 4
df['End_Rounded'] = (df['End_Hours'] * 4).round() / 4

schedule_summary = df.groupby(['Cashier ID', 'Day Name', 'Day Index']).agg({
    'Start_Rounded': 'median',
    'End_Rounded': 'median'
}).reset_index()

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

all_cashier_ids = sorted(schedule_summary['Cashier ID'].unique())

generated_files = []

for day in day_order:
    day_data = schedule_summary[schedule_summary['Day Name'] == day]
    if day_data.empty:
        continue
        
    day_data = day_data.sort_values('Cashier ID', ascending=False)
    
    plt.figure(figsize=(10, 8))
    
    y_labels = [str(int(cid)) for cid in day_data['Cashier ID']]
    widths = day_data['End_Rounded'] - day_data['Start_Rounded']
    lefts = day_data['Start_Rounded']
    
    plt.barh(y_labels, widths, left=lefts, color='skyblue', edgecolor='navy')
    
    plt.xlabel('Hour of Day (24h)')
    plt.ylabel('Cashier ID')
    plt.title(f'Daily Schedule: {day}')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.xlim(8, 22) 
    plt.xticks(range(8, 23))
    
    plt.tight_layout()
    filename = f'{day.lower()}_roster.png'
    plt.savefig(filename)
    generated_files.append(filename)
    plt.close()

print(f"Generated files: {generated_files}")