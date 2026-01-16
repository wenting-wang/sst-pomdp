import pandas as pd
import numpy as np
import glob
import os
import zipfile

# ================= CONFIGURATION =================
DATA_DIR = '/kyb/agpd/wwang/data/sst_valid_base'
OUTPUT_CSV = '/kyb/agpd/wwang/sst-pomdp/outputs/fast_rt_subjects.csv'
RT_THRESHOLD = 200  # Threshold for "suspiciously fast" in ms
# =================================================

def check_fast_responders():
    print(f"Scanning {DATA_DIR} for zip files...")
    zip_files = glob.glob(os.path.join(DATA_DIR, '*.zip'))
    print(f"Found {len(zip_files)} zip files. Analyzing for RT < {RT_THRESHOLD}ms...")

    suspicious_data = []

    for zip_path in zip_files:
        try:
            # Extract Subject ID
            base_name = os.path.basename(zip_path)
            # Logic to extract ID based on your filename patterns
            subject_id = base_name.split('_')[1] if 'NDAR_' in base_name else base_name.split('.')[0]

            with zipfile.ZipFile(zip_path, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('sst.csv') and '__MACOSX' not in f]
                if not csv_files:
                    continue
                
                with z.open(csv_files[0]) as f:
                    # Load minimal columns
                    df = pd.read_csv(f, usecols=['sst_primaryrt', 'sst_expcon'])

            # Filter 1: Valid Trials (RT > 0)
            valid_trials = df[df['sst_primaryrt'] > 0]
            total_valid_rt_count = len(valid_trials)

            if total_valid_rt_count == 0:
                continue

            # Filter 2: Suspiciously Fast Trials (0 < RT < 200)
            fast_trials = valid_trials[valid_trials['sst_primaryrt'] < RT_THRESHOLD]
            fast_count = len(fast_trials)

            # If they have ANY fast RTs, record them
            if fast_count > 0:
                suspicious_data.append({
                    'Subject': subject_id,
                    'Fast_RT_Count': fast_count,
                    'Total_Valid_RT_Trials': total_valid_rt_count,
                    'Pct_Fast_RT': (fast_count / total_valid_rt_count) * 100,
                    'Min_RT': fast_trials['sst_primaryrt'].min(),
                    'Mean_Fast_RT': fast_trials['sst_primaryrt'].mean()
                })

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    # --- Reporting ---
    if not suspicious_data:
        print("No subjects found with RT < 200ms.")
        return

    df_res = pd.DataFrame(suspicious_data)
    
    # Sort by count of fast RTs (descending) to see worst offenders first
    df_res = df_res.sort_values(by='Fast_RT_Count', ascending=False)

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_res.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*60)
    print(f"ANALYSIS COMPLETE: {len(df_res)} subjects had at least one RT < {RT_THRESHOLD}ms")
    print("="*60)
    print(f"Full list saved to: {OUTPUT_CSV}")
    print("\nTop 15 'Fast Responders' (High likelihood of invalid data):")
    print("-" * 60)
    print(f"{'Subject':<20} | {'Count < 200ms':<15} | {'% of Their Trials':<15}")
    print("-" * 60)
    
    # Print top 15 rows
    for index, row in df_res.head(15).iterrows():
        print(f"{row['Subject']:<20} | {int(row['Fast_RT_Count']):<15} | {row['Pct_Fast_RT']:.1f}%")
    print("-" * 60)

if __name__ == "__main__":
    check_fast_responders()