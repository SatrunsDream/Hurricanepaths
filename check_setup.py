import os

files_to_check = [
    'data/hurricane_paths_processed.pkl',
    'data/hurricane_paths_processed_MODEL.pkl',
    'data/sliding_results_final.pkl',
    'data/kalman_results_sliding_results_final.pkl',
    'data/val_sliding_results_final.pkl',
    'dashboard_data.csv'
]

print("=" * 60)
print("DASHBOARD SETUP CHECK")
print("=" * 60)
print("\nChecking required files:\n")

for f in files_to_check:
    exists = os.path.exists(f)
    status = "✓ EXISTS" if exists else "✗ MISSING"
    size = ""
    if exists:
        size_bytes = os.path.getsize(f)
        if size_bytes > 1024*1024:
            size = f" ({size_bytes/(1024*1024):.1f} MB)"
        elif size_bytes > 1024:
            size = f" ({size_bytes/1024:.1f} KB)"
        else:
            size = f" ({size_bytes} bytes)"
    print(f"  {status:12s} {f}{size}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Check if dashboard_data.csv exists
has_dashboard_data = os.path.exists('dashboard_data.csv')
has_track_data = os.path.exists('data/hurricane_paths_processed.pkl') or os.path.exists('data/hurricane_paths_processed_MODEL.pkl')
has_results = os.path.exists('data/sliding_results_final.pkl') or os.path.exists('data/kalman_results_sliding_results_final.pkl')

if has_dashboard_data:
    print("✓ dashboard_data.csv exists - Dashboard should work!")
else:
    print("✗ dashboard_data.csv MISSING - Need to run preprocessing")
    
if has_track_data:
    print("✓ Track data file exists - Map visualization will work")
else:
    print("✗ Track data file MISSING - Map visualization will not work")
    
if has_results:
    print("✓ Forecast results file exists")
else:
    print("✗ Forecast results file MISSING - Need to generate results first")

print("\n" + "=" * 60)

