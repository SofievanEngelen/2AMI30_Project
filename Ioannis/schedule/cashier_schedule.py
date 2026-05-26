import pm4py
import pandas as pd

log = pm4py.read_xes("Supermarket_data/Supermarket_Cashier.xes")
df = pm4py.convert_to_dataframe(log)

df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)

DAY_INDEX = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

def extract_cashier_schedule(df):
    records = []

    for cashier_id, group in df.groupby('case:concept:name'):
        group = group.sort_values('time:timestamp')

        starts = group[group['concept:name'] == 'Start Shift'].reset_index(drop=True)
        ends   = group[group['concept:name'] == 'End shift'].reset_index(drop=True)

        for i in range(min(len(starts), len(ends))):
            start_ts = starts.iloc[i]['time:timestamp']
            end_ts   = ends.iloc[i]['time:timestamp']
            day_name = start_ts.strftime('%A')

            records.append({
                'Cashier ID':     cashier_id,
                'Day Name':       day_name,
                'Day Index':      DAY_INDEX[day_name],
                'Start Time':     start_ts.strftime('%H:%M:%S'),
                'End Time':       end_ts.strftime('%H:%M:%S'),
                'Duration (min)': round((end_ts - start_ts).total_seconds() / 60, 1)
            })

    return (
        pd.DataFrame(records)
          .sort_values(['Cashier ID', 'Day Index'])
          .reset_index(drop=True)
    )


schedule = extract_cashier_schedule(df)
print(schedule.to_string(index=False))
schedule.to_csv('cashier_schedule.csv', index=False)