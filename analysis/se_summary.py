import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import zipfile

# ================= CONFIGURATION =================
# Path to your data folder
DATA_DIR = '/kyb/agpd/wwang/data/sst_valid_base'

# Output filenames
OUTPUT_CSV = '/kyb/agpd/wwang/sst-pomdp/outputs/se_summary.csv'
OUTPUT_PLOT_DIST = '/kyb/agpd/wwang/sst-pomdp/outputs/se_dist.png'

# Constants
STOP_SIGNAL_DURATION = 300  # ms
MODEL_HORIZON = 1000        # ms (1s)
# =================================================

def analyze_sst_errors_full():
    print(f"Scanning {DATA_DIR} for zip files...")
    zip_files = glob.glob(os.path.join(DATA_DIR, '*.zip'))
    print(f"Found {len(zip_files)} zip files.")

    results = []

    # Variables for Global Aggregation (Micro-Average)
    global_counts = {
        'Premature': 0,
        'Signal': 0,
        'Fixation': 0,
        'Late': 0,
        'Total_Errors': 0
    }

    for zip_path in zip_files:
        try:
            # Extract Subject ID
            base_name = os.path.basename(zip_path)
            subject_id = base_name.split('_')[1] if 'NDAR_' in base_name else base_name.split('.')[0]

            with zipfile.ZipFile(zip_path, 'r') as z:
                # Find the CSV inside the zip (ignore MACOSX)
                csv_files = [f for f in z.namelist() if f.endswith('sst.csv') and '__MACOSX' not in f]
                
                if not csv_files:
                    continue
                
                # Read the first valid CSV found
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f)

            # Filter for Stop Trials that are ERRORS (inhibitacc == 0)
            stop_errors = df[
                (df['sst_expcon'] == 'VariableStopTrial') & 
                (df['sst_inhibitacc'] == 0)
            ].copy()

            total_errors = len(stop_errors)
            if total_errors == 0:
                continue

            # Define Timepoints
            rt = stop_errors['sst_primaryrt']
            ssd = stop_errors['sst_ssd_dur']
            stop_signal_end = ssd + STOP_SIGNAL_DURATION

            # Count Errors by Type
            n_premature = np.sum(rt < ssd)
            n_signal = np.sum((rt >= ssd) & (rt <= stop_signal_end))
            n_fixation = np.sum((rt > stop_signal_end) & (rt <= MODEL_HORIZON))
            n_late = np.sum(rt > MODEL_HORIZON)

            # Update Global Counts
            global_counts['Premature'] += n_premature
            global_counts['Signal'] += n_signal
            global_counts['Fixation'] += n_fixation
            global_counts['Late'] += n_late
            global_counts['Total_Errors'] += total_errors

            # Store Per-Subject Results
            results.append({
                'Subject': subject_id,
                'Total_Stop_Errors': total_errors,
                'Premature': n_premature,
                'Signal': n_signal,
                'Fixation': n_fixation,
                'Late': n_late,
                'Pct_Premature': n_premature / total_errors,
                'Pct_Signal': n_signal / total_errors,
                'Pct_Fixation': n_fixation / total_errors,
                'Pct_Late': n_late / total_errors
            })

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    # --- Processing & Saving ---
    if not results:
        print("No valid data found.")
        return

    df_res = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['Subject', 'Total_Stop_Errors', 
            'Premature', 'Signal', 'Fixation', 'Late', 
            'Pct_Premature', 'Pct_Signal', 'Pct_Fixation', 'Pct_Late']
    df_res = df_res[cols]
    
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved individual subject data to {OUTPUT_CSV}")

    # --- Plotting ---
    
    # Plot 1: Distribution of Percentages (Boxplot)
    plt.figure(figsize=(8, 6))
    plot_data = [df_res['Pct_Premature'], df_res['Pct_Signal'], df_res['Pct_Fixation'], df_res['Pct_Late']]
    plt.boxplot(plot_data, labels=['Premature', 'Signal', 'Fixation', 'Late'])
    plt.title('Variability of Error Proportions (Per Subject)')
    plt.ylabel('Proportion')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(OUTPUT_PLOT_DIST)
    print(f"Saved distribution plot to {OUTPUT_PLOT_DIST}")

    # --- PRINT TOTAL STATISTICS ---
    print("\n" + "="*40)
    print("      GLOBAL STATISTICS (ALL SUBJECTS)      ")
    print("="*40)
    print(f"Total Subjects Processed: {len(df_res)}")
    print(f"Total Stop Errors Analyzed: {global_counts['Total_Errors']}")
    print("-" * 40)
    print(f"{'Condition':<15} | {'Count':<8} | {'Percent':<8}")
    print("-" * 40)
    
    for label in ['Premature', 'Signal', 'Fixation', 'Late']:
        count = global_counts[label]
        pct = (count / global_counts['Total_Errors']) * 100 if global_counts['Total_Errors'] > 0 else 0
        print(f"{label:<15} | {count:<8} | {pct:.2f}%")
    print("="*40)
    print(f"\nNote: 'Fixation' errors (SSD+300ms < RT <= 1s) are the critical late errors")
    print(f"that must be captured by your model.")

if __name__ == "__main__":
    analyze_sst_errors_full()


# ========================================
#       GLOBAL STATISTICS (ALL SUBJECTS)      
# ========================================
# Total Subjects Processed: 5918
# Total Stop Errors Analyzed: 162845
# ----------------------------------------
# Condition       | Count    | Percent 
# ----------------------------------------
# Premature       | 16032    | 9.84%
# Signal          | 128087   | 78.66%
# Fixation        | 17263    | 10.60%
# Late            | 1478     | 0.91%
# ========================================
