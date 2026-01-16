import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import zipfile

# ================= CONFIGURATION =================
DATA_DIR = '/kyb/agpd/wwang/data/sst_valid_base'
OUTPUT_PLOT = '/kyb/agpd/wwang/sst-pomdp/outputs/rt_histogram_clean.png'

# Cleaning Parameters
RT_CUTOFF_MS = 200          # Trials faster than this are "fast"
SUBJ_EXCLUDE_PCT = 1.0      # Drop subject if > 1.0% of their responses are fast
# =================================================

def plot_clean_distribution():
    print(f"Scanning {DATA_DIR}...")
    zip_files = glob.glob(os.path.join(DATA_DIR, '*.zip'))
    
    clean_rts = []
    stats = {
        'total_subjects': 0,
        'dropped_subjects': 0,
        'kept_subjects': 0,
        'total_trials_kept': 0
    }

    print(f"Processing {len(zip_files)} subjects...")
    print(f"Criteria: Drop subject if > {SUBJ_EXCLUDE_PCT}% responses are < {RT_CUTOFF_MS}ms")

    for zip_path in zip_files:
        stats['total_subjects'] += 1
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('sst.csv') and '__MACOSX' not in f]
                if not csv_files: continue
                
                with z.open(csv_files[0]) as f:
                    # Read only RT column
                    df = pd.read_csv(f, usecols=['sst_primaryrt'])

            # Get all responses (RT > 0)
            valid_responses = df[df['sst_primaryrt'] > 0]['sst_primaryrt'].values
            n_responses = len(valid_responses)
            
            if n_responses == 0:
                continue

            # Check Subject Quality
            n_fast = np.sum(valid_responses < RT_CUTOFF_MS)
            pct_fast = (n_fast / n_responses) * 100

            if pct_fast > SUBJ_EXCLUDE_PCT:
                stats['dropped_subjects'] += 1
                continue # SKIP BAD SUBJECT ENTIRELY

            # Subject is good: Keep their valid trials (RT >= 200)
            # We still remove the specific rare 'slip' trials < 200ms
            good_trials = valid_responses[valid_responses >= RT_CUTOFF_MS]
            
            if len(good_trials) > 0:
                clean_rts.append(good_trials)
                stats['kept_subjects'] += 1
                stats['total_trials_kept'] += len(good_trials)

        except Exception as e:
            print(f"Error reading {os.path.basename(zip_path)}: {e}")

    # --- Aggregation & Plotting ---
    if not clean_rts:
        print("No data left after cleaning.")
        return

    all_clean_rts = np.concatenate(clean_rts)
    
    print("\n" + "="*40)
    print("      CLEANING SUMMARY      ")
    print("="*40)
    print(f"Total Subjects Scanned: {stats['total_subjects']}")
    print(f"Subjects Dropped (>1% fast): {stats['dropped_subjects']} ({stats['dropped_subjects']/stats['total_subjects']:.1%})")
    print(f"Subjects Kept: {stats['kept_subjects']}")
    print(f"Total Clean Trials: {len(all_clean_rts)}")
    print(f"Mean RT (Cleaned): {np.mean(all_clean_rts):.2f} ms")
    print("="*40)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(all_clean_rts, bins=100, range=(0, 1500), 
             color='#2ca02c', edgecolor='black', alpha=0.7) # Green for 'Clean'
    
    plt.title(f'Cleaned Reaction Time Distribution\n(Subjects with <{SUBJ_EXCLUDE_PCT}% fast errors kept; Trials <{RT_CUTOFF_MS}ms removed)', fontsize=13)
    plt.xlabel('Reaction Time (ms)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    mean_rt = np.mean(all_clean_rts)
    plt.axvline(mean_rt, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean RT: {mean_rt:.0f}ms')
    plt.legend()

    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT)
    print(f"\nSaved clean plot to {OUTPUT_PLOT}")

if __name__ == "__main__":
    plot_clean_distribution()