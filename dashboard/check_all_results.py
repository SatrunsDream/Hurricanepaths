import os
import pandas as pd

files = [
    'data/kalman_results_sliding_results.pkl',
    'data/kalman_results_sliding_results_improved.pkl', 
    'data/kalman_results_sliding_results_final.pkl'
]

print("Checking all sliding results files:")
print("="*60)

for f in files:
    if os.path.exists(f):
        df = pd.read_pickle(f)
        col_name = 'storm_id' if 'storm_id' in df.columns else 'sid'
        storms = df[col_name].nunique()
        print(f"\n{f}:")
        print(f"  Storms: {storms}")
        print(f"  Forecast instances: {len(df)}")
        if storms > 0:
            sample_sids = list(df[col_name].unique()[:5])
            print(f"  Sample SIDs: {sample_sids}")
            # Check date range
            if 'storm_id' in df.columns:
                years = [int(sid[:4]) for sid in df[col_name].unique() if len(sid) >= 4]
                if years:
                    print(f"  Year range: {min(years)} to {max(years)}")
    else:
        print(f"\n{f}: NOT FOUND")

